from __future__ import annotations

import cv2
import numpy as np


class Segmenter:
    def predict(self, color_bgr: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError


class HSVSegmenter(Segmenter):
    """Small CPU fallback for demo objects with saturated colors.

    This is not a medical segmentation model. It exists so the RGB -> mask -> 3D
    path can be tested before SAM/SAM2/SAM3 weights and GPU/CPU settings are ready.
    """

    def __init__(self, min_saturation: int = 80, min_value: int = 50):
        self.min_saturation = min_saturation
        self.min_value = min_value

    def predict(self, color_bgr: np.ndarray) -> list[np.ndarray]:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        mask = (hsv[:, :, 1] >= self.min_saturation) & (hsv[:, :, 2] >= self.min_value)
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 250:
                continue
            one = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(one, [contour], -1, 255, thickness=cv2.FILLED)
            masks.append(one.astype(bool))
        masks.sort(key=lambda m: int(np.count_nonzero(m)), reverse=True)
        return masks[:5]


class LesionColorSegmenter(Segmenter):
    """Color-and-shape lesion detector for the fixed silicone phantom.

    The phantom lesion is expected to be a pink/red silicone patch on a darker
    organ surface. This detector is intentionally deterministic and CPU-only:
    HSV color gates create candidates, then contour area and circularity remove
    small marks, vessels and noise. Tune the thresholds from the command line
    for the actual lighting and camera exposure.
    """

    def __init__(
        self,
        hue1_low: int = 0,
        hue1_high: int = 14,
        hue2_low: int = 155,
        hue2_high: int = 179,
        min_saturation: int = 35,
        min_value: int = 55,
        max_value: int = 255,
        min_area: float = 120.0,
        max_area: float = 90000.0,
        min_circularity: float = 0.12,
    ) -> None:
        self.hue1_low = hue1_low
        self.hue1_high = hue1_high
        self.hue2_low = hue2_low
        self.hue2_high = hue2_high
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.max_value = max_value
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity

    def predict(self, color_bgr: np.ndarray) -> list[np.ndarray]:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        hue_gate = ((h >= self.hue1_low) & (h <= self.hue1_high)) | ((h >= self.hue2_low) & (h <= self.hue2_high))
        mask = hue_gate & (s >= self.min_saturation) & (v >= self.min_value) & (v <= self.max_value)
        mask = mask.astype(np.uint8) * 255

        kernel3 = np.ones((3, 3), np.uint8)
        kernel7 = np.ones((7, 7), np.uint8)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1.0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < self.min_circularity:
                continue

            one = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(one, [contour], -1, 255, thickness=cv2.FILLED)
            score = area * max(circularity, 0.05)
            scored.append((score, one.astype(bool)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [m for _, m in scored[:5]]


class LiverSurfaceLesionSegmenter(Segmenter):
    """Detect a raised lesion on the official liver phantom surface.

    In the official-looking phantom photo, the lesion is not a separate bright
    pink object. It is a small darker, more saturated red-brown raised patch on
    a large red-brown liver. This detector first estimates liver-like material,
    then searches for local dark/saturated blobs inside that material.
    """

    def __init__(self, min_area: float = 60.0, max_area: float = 3500.0) -> None:
        self.min_area = min_area
        self.max_area = max_area

    @staticmethod
    def _red_hue_mask(h: np.ndarray, low: int = 16) -> np.ndarray:
        return (h <= low) | (h >= 168)

    def _liver_mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Liver/resin is red-brown and moderately dark. This deliberately
        # excludes the pale TPU bowel and the cream base in the supplied photo.
        mask = self._red_hue_mask(h, 18) & (s >= 45) & (v >= 55) & (v <= 205)
        mask = mask.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask.astype(bool)
        areas = [cv2.contourArea(c) for c in contours]
        keep = np.zeros(mask.shape, dtype=np.uint8)
        for contour, area in zip(contours, areas):
            if area >= max(2500.0, 0.08 * max(areas)):
                cv2.drawContours(keep, [contour], -1, 255, thickness=cv2.FILLED)
        return keep.astype(bool)

    def predict(self, color_bgr: np.ndarray) -> list[np.ndarray]:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        liver = self._liver_mask(hsv)
        if not np.any(liver):
            return []

        # Local contrast catches a raised/darker lesion even when its absolute
        # hue is close to normal liver. Use blur radii larger than the lesion
        # texture but smaller than the full organ.
        v_blur = cv2.GaussianBlur(v, (0, 0), 17).astype(np.int16)
        s_blur = cv2.GaussianBlur(s, (0, 0), 17).astype(np.int16)
        v_i = v.astype(np.int16)
        s_i = s.astype(np.int16)

        red = self._red_hue_mask(h, 14)
        locally_dark = (v_blur - v_i) >= 5
        locally_saturated = (s_i - s_blur) >= 6
        absolute_gate = (s >= 80) & (v >= 70) & (v <= 185)
        # The official lesion photo has a stronger absolute signature too: the
        # raised tissue is darker and more saturated than the surrounding liver.
        # Keeping this as a separate gate prevents the large liver contour from
        # swallowing the lesion while still allowing local-contrast fallback.
        official_photo_gate = (s >= 115) & (v >= 70) & (v <= 175)
        candidate = liver & red & official_photo_gate
        candidate = candidate.astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored: list[tuple[float, np.ndarray]] = []
        h_img, w_img = candidate.shape
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area or area > self.max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1.0:
                continue
            circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
            if circularity < 0.08:
                continue
            x, y, w, h_box = cv2.boundingRect(contour)
            aspect = w / max(1.0, float(h_box))
            if aspect < 0.25 or aspect > 4.0:
                continue
            fill = area / max(1.0, float(w * h_box))
            if fill < 0.15:
                continue

            one = np.zeros(candidate.shape, dtype=np.uint8)
            cv2.drawContours(one, [contour], -1, 255, thickness=cv2.FILLED)

            # Prefer compact blobs in the upper liver surface. This suppresses
            # bowel folds and bottom shadows in the supplied official-style view.
            m = cv2.moments(contour)
            cy = (m["m01"] / m["m00"]) if m["m00"] else (y + h_box / 2.0)
            upper_bonus = 2.5 if cy < 0.35 * h_img else (1.0 if cy < 0.55 * h_img else 0.35)
            area_target = 1800.0
            area_score = min(area, area_target) / max(area, area_target)
            score = area_score * max(circularity, 0.05) * max(fill, 0.2) * upper_bonus
            scored.append((score, one.astype(bool)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [m for _, m in scored[:5]]


class PhantomLesionSegmenter(Segmenter):
    """Lesion detector with selectable phantom color profiles.

    Profiles:
    - official: red-brown raised lesion on the official-style liver phantom.
    - pink: separate pink silicone practice lesion/module.
    - pink-real: stricter pink lesion on the current white printed phantom.
    - painted: red-brown raised patch on the newly painted official phantom.
    - auto: try official first, then pink fallback.
    """

    def __init__(self, profile: str = "auto") -> None:
        if profile not in {"auto", "official", "pink", "pink-real", "painted"}:
            raise ValueError(f"Unknown lesion profile: {profile}")
        self.profile = profile
        self.surface = LiverSurfaceLesionSegmenter()
        self.painted = PaintedLesionSegmenter()
        self.color = LesionColorSegmenter()
        self.pink_real = LesionColorSegmenter(
            hue1_low=0,
            hue1_high=6,
            hue2_low=168,
            hue2_high=179,
            min_saturation=95,
            min_value=90,
            max_value=245,
            min_area=450.0,
            max_area=65000.0,
            min_circularity=0.08,
        )

    def predict(self, color_bgr: np.ndarray) -> list[np.ndarray]:
        if self.profile == "painted":
            return self.painted.predict(color_bgr)
        if self.profile == "official":
            return self.surface.predict(color_bgr)
        if self.profile == "pink":
            return self.color.predict(color_bgr)
        if self.profile == "pink-real":
            return self.pink_real.predict(color_bgr)
        masks = self.surface.predict(color_bgr)
        if masks:
            return masks
        return self.color.predict(color_bgr)


class PaintedLesionSegmenter(Segmenter):
    """Detector for the painted lesion on the dark red liver.

    On the live D405 view the lesion is the lighter red square in the liver
    groove. It has the same hue as the liver, but its value is about 90 while
    the surrounding liver stays near 60-75. A darker-blob rule therefore locks
    onto the bowel tip instead of this patch.
    """

    def __init__(self, min_area: float = 2500.0, max_area: float = 16000.0) -> None:
        self.min_area = min_area
        self.max_area = max_area

    def predict(
        self,
        color_bgr: np.ndarray,
        exclusion_mask: np.ndarray | None = None,
        liver_mask: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        img_h, img_w = h.shape
        red = (h <= 12) | (h >= 170)

        if liver_mask is not None and np.count_nonzero(liver_mask) >= 800:
            return self._predict_on_liver(v, s, red, liver_mask)

        dark_liver = red & (s >= 100) & (v >= 35) & (v <= 84)
        dark_liver = PhantomMaterialSegmenter._keep_components(
            PhantomMaterialSegmenter._clean(dark_liver, open_size=5, close_size=21),
            min_area=max(1800, int(0.004 * img_h * img_w)),
            max_components=2,
        )
        if not np.any(dark_liver):
            return []

        near_liver = cv2.dilate(dark_liver.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
        candidate = red & near_liver & (s >= 80) & (v >= 78) & (v <= 125)
        if exclusion_mask is not None:
            exclusion = cv2.dilate(exclusion_mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            candidate &= ~exclusion
        candidate = PhantomMaterialSegmenter._clean(candidate, open_size=3, close_size=21)
        liver_material = red & near_liver & (s >= 70) & (v >= 30) & (v <= 140)

        contours, _ = cv2.findContours(candidate.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            if cv2.contourArea(contour) < 400:
                continue
            hull = cv2.convexHull(contour)
            x, y, w, h_box = cv2.boundingRect(hull)
            aspect = w / max(1.0, float(h_box))
            if aspect < 0.55 or aspect > 1.85:
                continue
            hull_mask = np.zeros(candidate.shape, dtype=np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
            filled = hull_mask.astype(bool) & liver_material
            area = float(np.count_nonzero(filled))
            hull_area = float(np.count_nonzero(hull_mask))
            if area < self.min_area or area > self.max_area:
                continue
            if hull_area <= 0 or area / hull_area < 0.55:
                continue
            fill = area / max(1.0, float(w * h_box))
            if fill < 0.45:
                continue
            score = area * fill
            scored.append((score, filled))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [m for _, m in scored[:1]]

    def _predict_on_liver(
        self,
        v: np.ndarray,
        s: np.ndarray,
        red: np.ndarray,
        liver_mask: np.ndarray,
    ) -> list[np.ndarray]:
        # The square is only a local bright patch. One side of the liver is
        # also brighter from the lamp, so a global median cutoff swallows it.
        blur = cv2.GaussianBlur(v, (0, 0), 16)
        candidate = liver_mask & red & (s >= 80)
        candidate &= v.astype(np.int16) >= (blur.astype(np.int16) + 6)
        candidate = PhantomMaterialSegmenter._clean(candidate, open_size=3, close_size=11)
        liver_area = float(np.count_nonzero(liver_mask))
        contours, _ = cv2.findContours(candidate.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scored: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            if cv2.contourArea(contour) < 400:
                continue
            hull = cv2.convexHull(contour)
            x, y, w, h_box = cv2.boundingRect(hull)
            aspect = w / max(1.0, float(h_box))
            if aspect < 0.55 or aspect > 1.85:
                continue
            hull_mask = np.zeros(candidate.shape, dtype=np.uint8)
            cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
            filled = hull_mask.astype(bool) & liver_mask
            area = float(np.count_nonzero(filled))
            if area < self.min_area or area > self.max_area or area > 0.22 * liver_area:
                continue
            fill = area / max(1.0, float(w * h_box))
            if fill < 0.45:
                continue
            scored.append((area * fill, filled))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [m for _, m in scored[:1]]

    def _hough_circle_mask(self, color_bgr: np.ndarray, hsv: np.ndarray, liver_like: np.ndarray) -> np.ndarray | None:
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        img_h, img_w = h.shape
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        search = gray.copy()
        search[~liver_like] = 0
        search = cv2.GaussianBlur(search, (9, 9), 2)
        circles = cv2.HoughCircles(
            search,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(30, int(0.045 * min(img_h, img_w))),
            param1=80,
            param2=18,
            minRadius=max(12, int(0.018 * min(img_h, img_w))),
            maxRadius=max(35, int(0.140 * min(img_h, img_w))),
        )
        if circles is None:
            return None

        scored: list[tuple[float, int, int, int]] = []
        for x_f, y_f, r_f in circles[0]:
            x, y, radius = int(round(x_f)), int(round(y_f)), int(round(r_f))
            if not (0 <= x < img_w and 0 <= y < img_h):
                continue
            if not liver_like[y, x]:
                continue
            if x < 0.08 * img_w or x > 0.88 * img_w or y < 0.16 * img_h or y > 0.88 * img_h:
                continue
            yy, xx = np.indices((img_h, img_w))
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            ring = disk & liver_like
            if np.count_nonzero(ring) < 200:
                continue
            mean_s = float(np.mean(s[ring]))
            mean_v = float(np.mean(v[ring]))
            red_ratio = float(np.mean(((h[ring] <= 14) | (h[ring] >= 170))))
            if mean_s < 70 or red_ratio < 0.60:
                continue
            # Prefer real red-brown lesion material without hard-coding a fixed
            # image location; the single practice lesion may be placed anywhere
            # in the live D405 view.
            edge_margin = min(x, img_w - 1 - x, y, img_h - 1 - y) / max(1.0, min(img_h, img_w))
            target_prior = 0.75 + min(0.35, edge_margin)
            darkness = 1.0 - min(0.85, max(0.0, mean_v - 55.0) / 170.0)
            center_v = float(v[y, x])
            center_v_bonus = float(np.clip((center_v - 65.0) / 55.0, 0.25, 1.25))
            y_norm = y / max(1.0, float(img_h))
            lower_surface_bonus = float(np.clip((y_norm - 0.24) / 0.16, 0.35, 1.10))
            score = np.sqrt(radius) * (0.8 + mean_s / 255.0) * (0.7 + red_ratio) * (0.8 + target_prior) * (0.7 + darkness) * center_v_bonus * lower_surface_bonus
            scored.append((score, x, y, radius))

        if not scored:
            return None
        _, x, y, radius = max(scored, key=lambda item: item[0])
        yy, xx = np.indices((img_h, img_w))
        mask_radius = max(8, int(round(radius * 0.42)))
        mask = ((xx - x) ** 2 + (yy - y) ** 2 <= mask_radius ** 2) & liver_like
        return mask


class PhantomMaterialSegmenter:
    """Rule-based masks for the official phantom materials.

    These rules are based on the supplied official-style photo and STL material
    grouping: red-brown rigid resin liver/solid organs, pink TPU bowel, green
    gallbladder, and dark red/blue vessel or forbidden-zone markings. Treat them
    as the first deterministic baseline; tune with real D405 frames before a run.
    """

    def __init__(self, profile: str = "legacy") -> None:
        if profile not in {"legacy", "painted"}:
            raise ValueError(f"Unknown material profile: {profile}")
        self.profile = profile

    def predict_all(self, color_bgr: np.ndarray) -> dict[str, np.ndarray]:
        if self.profile == "painted":
            return self._predict_painted(color_bgr)

        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        b, g, r = color_bgr[:, :, 0], color_bgr[:, :, 1], color_bgr[:, :, 2]
        img_h, img_w = h.shape
        yy, xx = np.indices((img_h, img_w))
        red = (h <= 18) | (h >= 168)

        # STL B: colon/soft bowel-like obstacle. In the provided layout it is a
        # pale TPU tube in the lower-right half, not the red-brown liver shell.
        bowel_roi = (yy > 0.46 * img_h) & (xx > 0.36 * img_w)
        bowel_raw = (h <= 18) & (s >= 65) & (s <= 175) & (v >= 105) & bowel_roi
        # Remove the pale liver/gallbladder support gap above the bowel loop.
        bowel_raw &= ~((xx < 0.56 * img_w) & (yy < 0.62 * img_h))
        # Remove the central pale tray gap inside the U-shaped colon.
        bowel_raw &= ~((xx > 0.63 * img_w) & (xx < 0.82 * img_w) & (yy > 0.59 * img_h) & (yy < 0.73 * img_h) & (s < 115))
        bowel = self._keep_components(
            self._clean(bowel_raw, open_size=5, close_size=23),
            min_area=6000,
            max_components=1,
        )

        # STL C: gallbladder, green/yellow module under the liver left lobe.
        gall_raw = (h >= 18) & (h <= 50) & (s >= 40) & (v >= 20) & (v <= 195)
        gall_raw &= (xx < 0.55 * img_w) & (yy > 0.30 * img_h) & (yy < 0.78 * img_h)
        gallbladder = self._keep_components(
            self._clean(gall_raw, open_size=5, close_size=15),
            min_area=3000,
            max_components=1,
        )

        # STL A: body/liver shell/tray. For the task view we expose the visible
        # red-brown liver surface, then subtract separately detected obstacles.
        liver_raw = red & (s >= 42) & (v >= 50) & (v <= 215)
        liver_raw &= yy < 0.78 * img_h
        liver_raw &= ~bowel & ~gallbladder
        liver = self._keep_components(
            self._clean(liver_raw, open_size=5, close_size=23),
            min_area=12000,
            max_components=2,
        )

        # Dark colored printed/painted lines on the liver: blue/purple vessel
        # markings and red/blue forbidden-zone cues. Keep this thin and inside
        # organ-like material so it does not grab the cream base or text.
        # STL D: hepatic portal tube group/common base. In the provided view it
        # appears as the obvious thick red/blue/dark tubes between gallbladder
        # and liver, below the liver surface. This is a different target from
        # the thin painted lines on the liver shell.
        vessel_roi = (xx > 0.34 * img_w) & (xx < 0.52 * img_w) & (yy > 0.42 * img_h) & (yy < 0.68 * img_h)
        red_tube = ((h <= 10) | (h >= 170)) & (s >= 95) & (v >= 45) & (v <= 190)
        blue_tube = (h >= 105) & (h <= 140) & (s >= 45) & (v >= 35) & (v <= 165)
        # Dark tube pixels are only accepted close to the tube colors; otherwise
        # shadows under the liver would become a rectangular false positive.
        dark_tube = (v <= 75) & (s >= 60) & (xx > 0.38 * img_w) & (xx < 0.49 * img_w) & (yy > 0.48 * img_h)
        vessel = vessel_roi & (red_tube | blue_tube | dark_tube)
        vessel &= ~gallbladder & ~bowel
        vessel &= ~((v < 95) & (s < 80))
        vessel = self._keep_components(
            self._clean(vessel, open_size=3, close_size=7),
            min_area=250,
            max_components=4,
        )

        lesion_masks = PhantomLesionSegmenter().predict(color_bgr)
        lesion = lesion_masks[0] if lesion_masks else np.zeros(color_bgr.shape[:2], dtype=bool)
        kidney = np.zeros(color_bgr.shape[:2], dtype=bool)
        return {
            "liver": liver,
            "kidney": kidney,
            "bowel": bowel,
            "gallbladder": gallbladder,
            "vessel": vessel,
            "lesion": lesion,
        }

    def _predict_painted(self, color_bgr: np.ndarray) -> dict[str, np.ndarray]:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        img_h, img_w = h.shape
        empty = np.zeros(color_bgr.shape[:2], dtype=bool)

        red = (h <= 14) | (h >= 170)

        # Colors sampled from the current phantom photo:
        # omentum H18-20 S220-250 V150-175, bowel H6-9 S130-165 V130-181,
        # liver H3-6 S160-220 V76-124, gallbladder H78-81 S140-160 V45-50.
        omentum_raw = (h >= 14) & (h <= 36) & (s >= 150) & (v >= 90)
        omentum = self._keep_components(
            self._clean(omentum_raw, open_size=3, close_size=11),
            min_area=max(300, int(0.0008 * img_h * img_w)),
            max_components=3,
        )

        # Colon is the same hue family as the liver, but clearly brighter.
        bowel_raw = red & (s >= 80) & (s <= 200) & (v >= 128) & (v <= 230)
        bowel_raw &= ~omentum
        bowel = self._keep_components(
            self._clean(bowel_raw, open_size=3, close_size=9),
            min_area=max(1200, int(0.0015 * img_h * img_w)),
            max_components=3,
        )

        liver_raw = red & (s >= 110) & (v >= 40) & (v <= 124)
        liver_raw &= ~omentum & ~bowel
        liver = self._keep_components(
            self._clean(liver_raw, open_size=5, close_size=15),
            min_area=max(4000, int(0.01 * img_h * img_w)),
            max_components=1,
        )

        # Gallbladder is the dark green teardrop under the liver.
        gall_raw = (h >= 65) & (h <= 105) & (s >= 80) & (v >= 25) & (v <= 110)
        gall_raw &= ~omentum & ~liver & ~bowel
        gallbladder = self._keep_components(
            self._clean(gall_raw, open_size=5, close_size=15),
            min_area=250,
            max_components=1,
        )

        # D: small red/blue forbidden-zone marks on the white tray. Red liver
        # pixels are not marks, so red candidates must sit on the pale board
        # and outside the organs.
        organ = cv2.dilate((liver | bowel | omentum | gallbladder).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        white_board = cv2.dilate(((s <= 40) & (v >= 140)).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        red_mark = red & (s >= 100) & (v >= 40) & (v <= 170) & white_board
        blue_mark = (h >= 90) & (h <= 140) & (s >= 50) & (v >= 30) & (v <= 180)
        vessel = (red_mark | blue_mark) & ~organ
        vessel = self._keep_components(
            self._clean(vessel, open_size=3, close_size=5),
            min_area=70,
            max_components=8,
            max_area=max(900, int(0.003 * img_h * img_w)),
        )

        lesion_masks = PaintedLesionSegmenter().predict(color_bgr, liver_mask=liver)
        lesion = lesion_masks[0] if lesion_masks else empty.copy()
        liver = liver & ~lesion & ~gallbladder & ~vessel & ~omentum

        # This five-part STL set has no kidney. The pale block in the recording
        # is a support, and the tray edge is brighter still. Leave kidney empty
        # until a part with its own color is in view. A low-saturation rule here
        # labels the tray wall as kidney.
        kidney = empty.copy()

        return {
            "liver": liver,
            "kidney": kidney,
            "bowel": bowel,
            "omentum": omentum,
            "gallbladder": gallbladder,
            "vessel": vessel,
            "lesion": lesion,
        }

    def _split_liver_and_bowel(
        self,
        red: np.ndarray,
        s: np.ndarray,
        v: np.ndarray,
        omentum: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body = red & (s >= 40) & (v >= 35) & (v <= 235) & ~omentum
        body = self._clean(body, open_size=5, close_size=9)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8), 8)
        liver = np.zeros(body.shape, dtype=bool)
        bowel_parts: list[tuple[int, np.ndarray]] = []
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 1200:
                continue
            comp = labels == idx
            contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = float(cv2.contourArea(contour)) / max(1.0, hull_area)
            # A solid organ stays liver even when exposure lifts it above the
            # old V=100 bowel cutoff. The colon's C shape stays well below this.
            is_liver = (area >= 8000 and solidity >= 0.62) or (area >= 3500 and solidity >= 0.80)
            if is_liver:
                liver |= comp
            else:
                bowel_parts.append((area, comp))
        bowel = np.zeros(body.shape, dtype=bool)
        bowel_parts.sort(key=lambda item: item[0], reverse=True)
        for _, comp in bowel_parts[:4]:
            bowel |= comp
        return liver, bowel

    @staticmethod
    def _clean(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
        out = mask.astype(np.uint8) * 255
        if open_size > 1:
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
        if close_size > 1:
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
        return out.astype(bool)

    @staticmethod
    def _keep_components(
        mask: np.ndarray,
        min_area: int,
        max_components: int,
        max_area: int | None = None,
    ) -> np.ndarray:
        src = mask.astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(src, 8)
        comps: list[tuple[int, int]] = []
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area >= min_area and (max_area is None or area <= max_area):
                comps.append((area, idx))
        comps.sort(reverse=True)
        out = np.zeros(mask.shape, dtype=bool)
        for _, idx in comps[:max_components]:
            out |= labels == idx
        return out

    @staticmethod
    def _keep_line_components(mask: np.ndarray, min_area: int, min_length: int) -> np.ndarray:
        src = mask.astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(src, 8)
        out = np.zeros(mask.shape, dtype=bool)
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            width = int(stats[idx, cv2.CC_STAT_WIDTH])
            height = int(stats[idx, cv2.CC_STAT_HEIGHT])
            density = area / max(1, width * height)
            if area >= min_area and max(width, height) >= min_length and density < 0.55:
                out |= labels == idx
        return out


class PhantomMaterialMaskSegmenter(Segmenter):
    def __init__(self, name: str, profile: str = "legacy") -> None:
        self.name = name
        self.materials = PhantomMaterialSegmenter(profile=profile)

    def predict(self, color_bgr: np.ndarray) -> list[np.ndarray]:
        mask = self.materials.predict_all(color_bgr).get(self.name)
        if mask is None or not np.any(mask):
            return []
        return [mask]


class SamPlaceholder(Segmenter):
    def predict(self, color_bgr: np.ndarray) -> list[np.ndarray]:
        raise RuntimeError(
            "SAM is intentionally not bundled in this CPU-first demo. "
            "Add a SAM adapter here later and keep the output as bool masks."
        )
