# 工业相机导轨重建手册

程序是 `rail_stereo.py`。一台彩色工业相机沿导轨平移，拍两张静止场景，算出深度、点云，以及肝、结肠、网膜、胆囊、血管、病灶。类别用的是当前涂色体模的阈值，名字是 `painted`，不是 `pink-real`。`pink-real` 是以前白色肝脏上贴粉色病灶片用的。

深度公式：

```text
Z = f × B / d
```

`B` 是两张照片之间导轨实际走的距离，`d` 是视差（像素），`f` 是像素焦距。

## 现在能做到哪一步

和 D405 那张四联图比：

| 内容 | 工业相机 |
|---|---|
| 肝、结肠、网膜、胆囊、病灶圆 | 可以。一张彩色图就行，阈值和 D405 的 `painted` 相同 |
| 每一类的相机坐标 | 先不输出。`classes.json` 目前只记像素数 |
| 深度伪彩色和点云 | 可以。点云按照片上色，病灶点涂成红色 |
| 每帧实时更新 | 不行。场景必须静止，停稳拍两张 |
| 和 D405 一样稠密 | 还不行。光滑的肝脏和白底板容易是黑洞 |

黑白工业相机没有色相，这套阈值分不开肝和结肠。

## 拍摄

1. 用 MindVision 打开相机。对焦拧到拍体模的位置后锁死。两张图的焦距、光圈、曝光保持一致。
2. 相机平视，导轨上下或左右移动。镜头要横着切过导轨，不要顺着导轨前后推。前后推只有缩放，算不出深度。
3. 体模不要动。先拍 A。导轨走一段能读数的距离，体模大约在 15–30 cm 时先用 20–40 mm。再拍 B。
4. 竖直导轨：B 是相机在画面里向下移动之后的那张。左右导轨：B 是向右移动之后的那张。

照片放进 `captures/`，文件名自定。下面用 `rail_a.png`、`rail_b.png`。

## 不拍棋盘格时的焦距

工业相机不会在图像里带好 `fx、fy`。不标定时用镜头和规格书上的标称值：

```text
fx = 镜头焦距(mm) / 像元尺寸(mm)
fy = fx
```

像元 3.45 μm 写成 0.00345 mm。镜头写 12 mm 时，`fx = 12 / 0.00345 ≈ 3478`。光心不填，程序用图像中心。

焦距标称值偏几个百分点，整幅深度按同样比例偏。物体在 200 mm 处、焦距偏 5% 时，深度大约偏 10 mm。要收掉这个比例，再对锁死的对焦拍棋盘格，见文末。

## 只看类别

还没有第二张图时，先确认颜色分得开：

```powershell
python rail_stereo.py segment --image captures/rail_a.png --out-dir captures/rail_classes
```

看 `captures/rail_classes/classes.png`。图上只有类别名，没有像素坐标。`classes.json` 里只有每一类的 `pixels`。

## 两张图重建

竖直导轨：

```powershell
python rail_stereo.py reconstruct --image-a captures/rail_a.png --image-b captures/rail_b.png --baseline-mm 30 --fx 3478 --fy 3478 --out-dir captures/rail_recon
```

左右导轨加上 `--axis horizontal`。有效像素很少时加上 `--swap`，表示相机实际移动方向和上面的约定相反。

`--fx 3478` 换成自己的镜头和像元算出来的数。`--baseline-mm` 填导轨刻度，不要估。

## 输出

都在 `--out-dir` 里。

| 文件 | 内容 |
|---|---|
| `preview.png` | 从左到右：类别图、第一张照片、深度伪彩色。黑色是没匹配上 |
| `classes.png` | 类别轮廓和名字。不标注像素坐标。病灶圆的半径等于掩膜面积 |
| `classes.json` | 每一类的 `pixels`。像素坐标和相机坐标先不写 |
| `depth_mm.png` | 16 位深度，单位毫米，0 表示无效 |
| `depth_m.npy` | 米制深度 |
| `cloud.ply` | 点云，单位米。可以用 MeshLab 或 Open3D 打开 |
| `summary.json` | 有效像素、整幅深度中位数，以及同样的类别结果 |

`classes.json` 里每一类目前只有 `pixels`，0 表示没找到。肾在这套五件体模里没有对应颜色，会一直是 0。像素坐标和相机坐标先不输出。

## 结果不对时

- 类别把肝和结肠并到一起，或病灶跑到亮边上：工业相机白平衡和 D405 不一样，把一张实拍发回来重新吸色。不要改成 `pink-real`。
- `preview.png` 右边几乎全黑：先加 `--swap`。仍全黑就加大 `--num-disparities`，例如 `256`。视差搜索范围要盖住 `d = f × B / Z`。`f = 3478`、`B = 30 mm`、`Z = 200 mm` 时，视差大约是 522 像素，默认 128 不够，这时应把基线改小，或把 `--num-disparities` 提高到不小于该视差的 16 的倍数。
- 深度整体偏大或偏小：检查 `--fx` 和 `--baseline-mm`。这两个数按比例直接乘到深度上。
- 肝脏中间是洞：树脂表面太平，两张图对不上。先保留有纹理的结肠、网膜。以后电机可以在 0、15、30、45 mm 各停一张再融合；仍然大片空洞时，只在拍深度的那几张打开散斑投影，分类那张保持投影关闭。

## 棋盘格内参

只有需要把标称焦距的比例误差收掉时才做。打印 `棋盘格标定板_7x7_20mm.svg`，比例 100%，不要“适合页面”。整板 140×140 mm，每格 20 mm，程序数的是 6×6 个内角点。板子贴平。对焦已经锁死之后，用这台工业相机拍大约 15 张：有的正对，有的倾斜，有的靠近画面四角，格子都要在画面里。

```powershell
python rail_stereo.py calibrate --images captures/rail_calib --square-mm 20 --out captures/rail_intrinsics.json
```

重投影误差低于 0.5 像素就可以用。重建时换成：

```powershell
python rail_stereo.py reconstruct --image-a captures/rail_a.png --image-b captures/rail_b.png --baseline-mm 30 --intrinsics captures/rail_intrinsics.json --out-dir captures/rail_recon
```

改了对焦或分辨率之后，这份内参失效，要重标。
