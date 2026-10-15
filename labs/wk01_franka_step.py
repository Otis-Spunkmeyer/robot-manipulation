"""Week 1 lab: load Franka Panda from MuJoCo Menagerie, step it, measure sim speed and hold error."""
import time
from pathlib import Path

import mujoco
import numpy as np

# Path to the Panda scene (robot + floor + light) in the Menagerie clone
SCENE = Path.home() / "dev" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"

# MjModel = constant description (geometry, masses, joints, actuators, timestep)
# MjData  = changing state (qpos, qvel, ctrl, contacts, time)
model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)

print(f"timestep: {model.opt.timestep * 1e3:.3f} ms")
print(f"nq={model.nq} (position coords)  nv={model.nv} (velocity coords)  nu={model.nu} (actuators)")
print("joints:   ", [model.joint(i).name for i in range(model.njnt)])
print("actuators:", [model.actuator(i).name for i in range(model.nu)])

# Start at the model's 'home' keyframe and command the actuators to hold it
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = model.key_ctrl[0]
q0 = data.qpos[:7].copy()  # 7 arm joints; the last 2 qpos entries are the fingers

# Step the physics N times and time it
N = 10_000
t0 = time.perf_counter()
for _ in range(N):
    mujoco.mj_step(model, data)
wall = time.perf_counter() - t0
sim_t = N * model.opt.timestep

print(f"\nsimulated {sim_t:.2f} s in {wall:.3f} s wall")
print(f"real-time factor: {sim_t / wall:.0f}x   ({N / wall:,.0f} steps/s)")

# How far did each joint sag while 'holding' home under gravity?
err_deg = np.degrees(data.qpos[:7] - q0)
print("hold error per joint (deg):", np.round(err_deg, 4))
print(f"max |hold error| = {np.abs(err_deg).max():.4f} deg")
