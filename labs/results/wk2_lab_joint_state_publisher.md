# Wk 2 Lab: SO-101 joint-state publisher (Python vs C++) and TF tree

Date: 2026-10-21. Machine: RTX 4070 desktop, Ubuntu 24.04, ROS 2 Jazzy, idle except RViz.

## Setup

Workspace `ros2_ws/src`:

| Package | Type | Contents |
|---|---|---|
| `so101_description` | ament_cmake (data) | URDF with `package://` meshes, 18 STL, `display.launch.py` (args `gui`, `world`, `rviz_config`), saved RViz config |
| `so101_nodes_py` | ament_python | `joint_wave` (JointState publisher, wave/hold), `static_world` (world -> base_link static TF) |
| `so101_nodes_cpp` | ament_cmake | `joint_wave` (C++ port, same parameters) |

Build: `colcon build --symlink-install`, 3 packages, 0 warnings (`-Wall -Wextra -Wpedantic`).

## 1. Publish timing, Python vs C++

Measured with `ros2 topic hz /joint_states` (receive-side timing, ~25 s windows at 50 Hz, 5000 msgs at 1 kHz). CPU from `top`, % of one core.

| Rate | Metric | Python | C++ |
|---|---|---|---|
| 50 Hz | mean rate (Hz) | 49.999 | 49.999 |
| 50 Hz | std dev of period (ms) | 0.09 | 0.10 |
| 50 Hz | min / max period (ms) | 19 / 21 | 20 / 21 |
| 1 kHz | mean rate (Hz) | 1000.002 | 1000.002 |
| 1 kHz | std dev of period (ms) | 0.06 | 0.05 |
| 1 kHz | min / max period (ms) | 0 / 2 | 0 / 2 |
| 1 kHz | CPU (% of one core) | 47 | 7.6 |

Pass criterion 50.0 +/- 0.5 Hz: both pass.

**Findings**

- Idle machine: no timing difference between languages up to 1 kHz. Measured jitter (~0.1 ms) is OS scheduler wake-up + DDS transport + the `ros2 topic hz` subscriber itself; the language only affects the callback, which is microseconds.
- Cost differs: Python uses ~6x the CPU of C++ for the same empty 1 kHz loop. That headroom is what keeps C++ jitter tails short under load.
- `min 0 / max 2 ms` at 1 kHz (ms-rounded): occasional late tick followed by an on-time tick; at least one full period missed in each run.
- Instrument limit: receive-side timing includes transport jitter. Publisher-only timing would use `header.stamp` differences.

## 2. TF chain world -> base_link -> tool

`static_world`: world -> base_link at xyz (0.10, 0.20, 0.0) m, yaw 90 deg. Joint pose (hold): (30, 20, -40, 15, 0) deg.

| Check | Expected | tf2_echo | Pass |
|---|---|---|---|
| world -> base_link translation (m) | (0.100, 0.200, 0.000) | (0.100, 0.200, 0.000) | yes |
| world -> base_link yaw (deg) | 90 | 90.000 | yes |
| world -> gripper_frame_link (m) | (0.2898, 0.5675, 0.2698) | (0.290, 0.567, 0.270) | yes (< 0.5 mm, display rounding) |
| R rows | Rz(90) R_base: rows (-r2, r1, r3) | match to 3 decimals | yes |
| `/tf_static` durability | TRANSIENT_LOCAL | TRANSIENT_LOCAL | yes |

Expected tool point: p_base from `fk.urdf_poe` = (0.3675, -0.1898, 0.2698) m; Rz(90) maps (x, y, z) -> (-y, x, z); plus t.

## Issues found and fixed

- rclpy in Jazzy raises `ExternalShutdownException` on Ctrl+C, not `KeyboardInterrupt`: both Python nodes catch it.
- `--symlink-install` links files present at build time; a new file (saved RViz config) needs a package rebuild.
- `rate_hz` is startup-only (timer period fixed at creation); other parameters re-read every tick.
- C++ `create_wall_timer` uses the OS clock; Python `create_timer` follows `use_sim_time`. Use `create_timer(this, get_clock(), ...)` for sim.
