from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create review-only YOLO-seg circle proposals for the central raised lesion regions."
    )
    parser.add_argument("--images", required=True, help="Directory of selected source images")
    parser.add_argument("--out-dir", required=True, help="Output directory for draft labels and overlays")
    parser.add_argument("--negative-ids", default="168-180", help="Comma-separated candidate IDs/ranges confirmed negative")
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--circle-sensitivity", type=int, default=9, help="Hough accumulator threshold; lower adds proposals")
    parser.add_argument("--mask-scale", type=float, default=0.86, help="Scale proposal radius to avoid including the outer ring")
    return parser.parse_args()


def parse_ids(spec: str) -> set[int]:
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = (int(value) for value in part.split("-", 1))
            ids.update(range(first, last + 1))
        else:
            ids.add(int(part))
    return ids


def candidate_id(path: Path) -> int | None:
    match = re.search(r"_(\d+)_f\d+", path.stem)
    return int(match.group(1)) if match else None


def find_circle_proposals(image: np.ndarray, sensitivity: int, mask_scale: float) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = (((hue <= 20) | (hue >= 160)) & (saturation >= 35) & (value <= 175)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    height, width = red.shape
    proposals: list[tuple[float, float, float, float]] = []

    for component in range(1, count):
        x, y, box_w, box_h, area = (int(value) for value in stats[component])
        short_side, long_side = min(box_w, box_h), max(box_w, box_h)
        aspect = long_side / max(1.0, float(short_side))
        component_pixels = labels[y:y + box_h, x:x + box_w] == component
        median_value = float(np.median(value[y:y + box_h, x:x + box_w][component_pixels]))
        if area < 700 or area > 16000 or short_side < 32 or long_side > 170 or aspect > 2.25 or median_value > 135:
            continue

        crop = image[y:y + box_h, x:x + box_w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.4)
        min_radius = max(6, int(round(short_side * 0.13)))
        max_radius = max(min_radius + 1, int(round(short_side * 0.40)))
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=max(10, int(round(short_side * 0.38))),
            param1=55,
            param2=sensitivity,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue

        options = []
        for cx, cy, radius in circles[0]:
            if not (0.22 * box_w <= cx <= 0.78 * box_w and 0.22 * box_h <= cy <= 0.78 * box_h):
                continue
            global_x = int(round(cx + x))
            global_y = int(round(cy + y))
            if not (0 <= global_x < width and 0 <= global_y < height and labels[global_y, global_x] == component):
                continue
            if saturation[global_y, global_x] < 45 or value[global_y, global_x] > 145:
                continue
            expected_radius = 0.20 * short_side
            radius_error = abs(radius - expected_radius) / max(1.0, expected_radius)
            score = float(radius_error)
            options.append((score, float(cx + x), float(cy + y), float(radius)))
        proposals.extend(options)

    proposals.sort(key=lambda item: item[0])
    kept: list[np.ndarray] = []
    centers: list[tuple[np.ndarray, float]] = []
    for _, cx, cy, radius in proposals:
        center = np.array([cx, cy], dtype=np.float64)
        if any(np.linalg.norm(center - existing) < 0.90 * min(radius, old_radius) for existing, old_radius in centers):
            continue
        radius *= mask_scale
        if radius < 4:
            continue
        angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        polygon = np.column_stack((cx + radius * np.cos(angles), cy + radius * np.sin(angles)))
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        kept.append(polygon)
        centers.append((center, radius))

    if len(kept) <= 1:
        return kept
    # The lesion samples appear as a spatial cluster on the table. Isolated
    # reddish tools/props should not become labels merely because they look round.
    point_centers = [np.mean(polygon, axis=0) for polygon in kept]
    unseen = set(range(len(kept)))
    clusters: list[list[int]] = []
    while unseen:
        seed = unseen.pop()
        cluster = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            neighbors = [
                idx for idx in unseen
                if np.linalg.norm(point_centers[current] - point_centers[idx]) <= 145.0
            ]
            for idx in neighbors:
                unseen.remove(idx)
                cluster.append(idx)
                stack.append(idx)
        clusters.append(cluster)
    largest = max(clusters, key=len)
    return [kept[index] for index in largest]


def yolo_row(class_id: int, polygon: np.ndarray, width: int, height: int) -> str:
    normalized = polygon.copy()
    normalized[:, 0] /= width
    normalized[:, 1] /= height
    values = " ".join(f"{value:.6f}" for value in normalized.reshape(-1))
    return f"{class_id} {values}"


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images)
    out_dir = Path(args.out_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if args.class_id < 0 or not 0.5 <= args.mask_scale <= 1.1:
        raise ValueError("class-id must be non-negative and mask-scale must be in [0.5, 1.1]")
    negatives = parse_ids(args.negative_ids)
    paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise RuntimeError(f"No images found in {images_dir}")

    labels_dir = out_dir / "labels"
    overlays_dir = out_dir / "overlays"
    labels_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    review_rows = []
    total_instances = 0

    for image_path in paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            review_rows.append([image_path.name, "unreadable", 0, "image_read_failed"])
            continue
        height, width = image.shape[:2]
        cid = candidate_id(image_path)
        if cid in negatives:
            polygons: list[np.ndarray] = []
            status = "user_confirmed_negative"
        else:
            polygons = find_circle_proposals(image, args.circle_sensitivity, args.mask_scale)
            status = "draft_review" if polygons else "needs_manual_annotation"

        label_path = labels_dir / f"{image_path.stem}.txt"
        rows = [yolo_row(args.class_id, polygon, width, height) for polygon in polygons]
        label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

        overlay = image.copy()
        for index, polygon in enumerate(polygons, start=1):
            points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
            center = tuple(np.round(np.mean(polygon, axis=0)).astype(int))
            cv2.putText(overlay, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(
            overlay, f"{status} instances={len(polygons)}", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.imwrite(str(overlays_dir / f"{image_path.stem}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])
        review_rows.append([image_path.name, status, len(polygons), label_path.name])
        total_instances += len(polygons)

    review_path = out_dir / "review.csv"
    with review_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "status", "proposal_instances", "label_file"])
        writer.writerows(review_rows)

    print(f"Images: {len(paths)}")
    print(f"Draft lesion instances: {total_instances}")
    print(f"Labels: {labels_dir}")
    print(f"Overlays: {overlays_dir}")
    print(f"Review table: {review_path}")
    print("All positive labels are unverified circle proposals; inspect and correct them before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
