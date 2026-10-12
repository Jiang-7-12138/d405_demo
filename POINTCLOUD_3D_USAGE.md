# D405 三维点云使用说明

## 1. 安装可选三维依赖

基础 demo 不强制依赖 Open3D。若需要真正可旋转的三维点云窗口，先安装：

```powershell
cd D:\HuaweiMoveData\Users\蒋高韵\Desktop\D405_vision_demo
python -m pip install -r requirements-3d.txt
```

## 2. 实时显示真正的 3D 点云

推荐先用这个命令：

```powershell
python run_demo.py --mask-mode lesion --lesion-profile pink-real --rs-filters --open3d --pc-stride 1
```

如果电脑比较卡，可以把 `--pc-stride 1` 改成 `--pc-stride 2` 或 `--pc-stride 4`。

如果点云毛刺很多，可以加上 Open3D 清理：

```powershell
python run_demo.py --mask-mode lesion --lesion-profile pink-real --rs-filters --open3d --open3d-clean --pc-stride 1
```

参数含义：

- `--rs-filters`：启用 RealSense 空间滤波、时间滤波和孔洞填补。
- `--open3d`：打开真正的可旋转三维点云窗口。
- `--open3d-clean`：对显示/保存的点云做体素降采样和离群点剔除。
- `--pc-stride 1`：每个有效深度像素都进入点云，显示更细，但更慢。

## 3. 保存并离线查看点云

运行 `run_demo.py` 时按 `s`，程序会保存：

- `*_color.png`：彩色图
- `*_depth_vis.png`：深度伪彩色图
- `*.npz`：原始彩色图和深度图
- `*_cloud.ply`：完整视野点云
- `*_<mode>_target_cloud.ply`：当前病灶或 ROI 掩码对应的目标点云

离线打开 PLY：

```powershell
python view_ply_open3d.py captures\frame_xxxxx_cloud.ply --clean
```

只看病灶/ROI 目标点云时，打开 `*_target_cloud.ply`。

## 3.1 主界面的 3D reconstruction preview

主 OpenCV 窗口现在会显示四块内容：

- 相机 RGB 原始画面；
- 深度伪彩色图；
- X-Z / X-Y 米制点云投影；
- `3D reconstruction results` 三维重建预览，带 XYZ 坐标轴和米制参考网格。

三维预览使用的仍然是 D405 反投影得到的相机坐标系：

```text
X = 向右，Y = RealSense 相机坐标中的向下，Z = 向前
```

只在显示时把 Y 轴翻到向上，方便肉眼理解。原始测量坐标、CSV 日志、机械臂坐标变换和保存的 PLY 点云不受影响。

推荐运行：

```powershell
python run_demo.py --rs-filters --pc-stride 1 --pc-color-mode rgb --open3d
```

如果想把三维预览改成按深度伪彩色着色：

```powershell
python run_demo.py --rs-filters --pc-stride 1 --pc-color-mode depth --open3d
```

## 4. 单视角和环绕扫描的区别

单视角 D405 可以重建相机看得到的正面表面，并足够用于病灶三维中心定位。

如果要得到物体背面、侧面或完整封闭模型，单视角做不到，需要相机环绕扫描或物体转动，并且必须记录每一帧相机位姿后进行多帧点云融合。

比赛定位阶段建议优先使用单视角：识别病灶二维掩码，读取掩码内深度，计算病灶三维中心，再转换到机械臂基坐标系。
