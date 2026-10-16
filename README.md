# D405 深度视觉 Demo

这是一个面向“具身智能微创辅助操作赛”的本机 RGB-D demo。当前目标不是训练模型，而是先把 D405 的 `RGB -> Depth -> Camera 3D -> 临时 World 3D` 链路跑通。

## 已实现

- 启动 Intel RealSense D405，读取并对齐 RGB / Depth。
- 显示彩色图和深度伪彩色图。
- 显示采样点云的俯视 X-Z 和正视 X-Y 投影；当前 ROI/mask 对应点云会用红色高亮。
- 鼠标悬停任意像素，实时输出 `depth + Camera XYZ`。
- 鼠标拖框生成 ROI mask，并把 ROI 内有效深度点反投影到 3D，输出稳健中心点。
- CPU 版 HSV 简单分割，用于测试 `mask + depth -> 3D centroid` 链路。
- 可选 ArUco 15 mm 临时标定，模拟比赛 `World` 坐标系，输出 `T_world_camera` 后的 World XYZ。
- 保存彩色图、深度图、`.npz` 原始帧和 CSV 测量记录。

## 环境

本机已检测到：

- Python 3.11
- `pyrealsense2`
- `opencv-python`
- `numpy`
- Intel RealSense D405

如果换电脑运行，可在本目录执行：

```powershell
python -m pip install -r requirements.txt
```

## 运行

进入目录：

```powershell
cd D:\HuaweiMoveData\Users\蒋高韵\Desktop\D405_vision_demo
```

启动实时 demo：

```powershell
python run_demo.py
```

如果目标距离更远或更近，可以调整 ROI/mask 统计用的有效深度范围：

```powershell
python run_demo.py --min-depth-mm 30 --max-depth-mm 600
```

生成临时 ArUco 标定图：

```powershell
python run_demo.py --make-marker
```

会生成 `aruco_marker_id0.png`。打印时把黑色 marker 的实际边长控制为 15 mm。然后运行：

```powershell
python run_demo.py --aruco --marker-size-mm 15
```

## 操作

- 鼠标移动：查看当前像素深度和 Camera XYZ。
- 鼠标左键拖框：生成 ROI，计算 ROI 三维中心。
- `h`：切换 HSV 自动 mask。适合先用彩色小物体测试 mask 到 3D。
- `c`：清除 ROI / mask。
- `a`：开关 ArUco 临时 World 标定。
- `s`：保存当前帧到 `captures/`，包括彩色图、深度图、`.npz` 和 `.ply` 点云。
- `r`：把当前 ROI/mask 的 3D 结果写入 `captures/measurements.csv`。
- `q` 或 `Esc`：退出。

## 坐标说明

点击像素 `(u, v)` 后，程序按 D405 内参反投影：

```text
Z = depth(u, v)
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

这里的 `[X, Y, Z]` 属于 RealSense 的 Camera frame，单位是米。启用 `--aruco` 后，程序用 15 mm ArUco marker 模拟赛场标定基准块，并把 mask 中心从 Camera frame 转到临时 World frame。

正式比赛的 15 mm × 15 mm 标定基准块若不是 ArUco，需要根据实物图案替换 `calibration.py` 里的检测逻辑，但后续的 `T_world_camera` 和坐标变换接口可以保持不变。

## 文件结构

- `camera_d405.py`：D405 启动、RGB/Depth 对齐、内参读取。
- `pointcloud.py`：像素/ROI/mask 到 Camera XYZ 和稳健中心点。
- `segmenter.py`：分割接口；当前提供 HSV CPU fallback，后续可接 SAM/SAM2/SAM3。
- `calibration.py`：ArUco 临时标定、PnP、坐标变换。
- `pointcloud_viewer.py`：点云投影显示和 `.ply` 导出。
- `visualization.py`：mask、文字、拼图显示。
- `run_demo.py`：主程序。
- `endoscope_viewer.py`：USB 内窥镜实时查看、截图和录像。

## 内窥镜实时查看

本机检测到两个 OpenCV 视频设备，其中 `index 1` 更像 Bellysim 内窥镜，分辨率为 800×600。

```powershell
python endoscope_viewer.py --index 1
```

如果画面不对，试试：

```powershell
python endoscope_viewer.py --index 0
```

按键：

- `s`：保存截图到 `captures/`。
- `r`：开始/停止录像。
- `m`：开关器官和病灶识别。
- `q` 或 `Esc`：退出。

病灶圆顶有三套颜色，不能混用。不写 `--profile` 时用的是亮红色那套。

浅粉色圆顶，比周围方块更浅、更亮（S 45–115，V 165–220）：

```powershell
python endoscope_viewer.py --index 1 --profile endoscope
```

亮红色圆顶（S 80–165，V 165–240）：

```powershell
python endoscope_viewer.py --index 1 --profile endoscope-bright
```

深红色圆顶，外圈又暗又饱和（S ≥ 165，V 70–120），中间有一块稍亮的圆斑：

```powershell
python endoscope_viewer.py --index 1 --profile endoscope-dark
```

圆心会停住，小幅晃动不会跟着改坐标。画面里的圆顶换了颜色，就换上面的命令。

## 下一步

1. 用桌面物体测试 D405 在 7-30 cm 的有效深度比例和重复定位方差。
2. 打印 15 mm ArUco，跑通 Camera 到临时 World 的标定方向、单位和轴向。
3. 接入 SAM 时，只需要让新的 segmenter 输出 `bool mask` 列表，后面的点云和坐标变换不用改。
4. 腔体和机械臂到位后，增加真实标定块检测和 `Robot Base <-> World` 标定接口。
