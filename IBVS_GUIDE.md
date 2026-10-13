# Point IBVS Integration

`ibvs_controller.py` adds a translation-only, image-based visual-servo control
calculation for a fixed D405 camera. It is deliberately a pure computation
module: it does not connect to a robot or send motion commands.

## Inputs and output

Each update needs:

- `tool_pixel`: current image location of the moving tool tip, in pixels.
- `target_pixel`: desired feature location, for example the center of the
  detected lesion mask.
- `tool_depth_m`: current depth of the tool-tip feature, not the lesion depth.
- D405 color intrinsics (`fx`, `fy`, `cx`, `cy`).
- `T_base_camera`: calibrated 4x4 transform from the D405 camera frame to the
  robot base frame, in meters.

The result contains bounded camera-frame and robot-base-frame linear velocity
references plus a validity flag and stop reason. The default speed limit is
5 mm/s, and features more than 120 pixels apart are rejected rather than
approached blindly. These are conservative software defaults for offline
development, not validated competition settings.

```python
from ibvs_controller import PointIBVSController

controller = PointIBVSController()
command = controller.compute(
    tool_pixel=(tool_u, tool_v),
    target_pixel=(lesion_u, lesion_v),
    tool_depth_m=tool_z_m,
    fx=intrinsics.fx,
    fy=intrinsics.fy,
    cx=intrinsics.cx,
    cy=intrinsics.cy,
    t_base_camera=t_base_camera,
)
if command.valid:
    # Hand command.base_linear_m_s to a separately validated robot adapter.
    pass
else:
    # Stop and report command.reason.
    pass
```

## Remaining integration work

The current vision demo detects the lesion and computes its camera-frame 3D
position, but it does not detect/track the tool tip or connect to a robot
controller. A real IBVS loop therefore still needs a reliable tool-tip image
feature (detected directly or projected from robot state and calibrated
transforms), current tool-tip depth, a measured `T_base_camera`, and the robot
vendor/ROS interface. A low-level adapter must apply robot Jacobian/kinematic
limits, workspace and collision checks, force limits, watchdogs, and an
independent emergency stop. Do not send these references to hardware until
coordinate conventions and sign are verified at very low speed in free space.

Run offline unit tests with:

```powershell
python -m unittest test_ibvs_controller.py
```
