"""Week 1 lab, part 2: predict P-servo hold error e_ss = -tau_g / kp, then compare to simulation."""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path.home() / "dev" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
ARM = 7  # arm joints; fingers excluded

# --- Actuator parameters (MuJoCo 'general' actuator: force = gain*ctrl + bias0 + bias1*length + bias2*velocity)
kp = model.actuator_gainprm[:ARM, 0]          # position gain [N*m/rad]
gear = model.actuator_gear[:ARM, 0]           # transmission ratio actuator -> joint
jnt = model.actuator_trnid[:ARM, 0]           # which joint each actuator drives
assert np.all(jnt == np.arange(ARM)), "expected actuator i to drive joint i"

# --- Reference pose: the 'home' keyframe and its commanded positions
mujoco.mj_resetDataKeyframe(model, data, 0)
ctrl_home = model.key_ctrl[0].copy()
q_ref = ctrl_home[:ARM] / gear

# --- Gravity torque at the reference pose: with qvel = 0, qfrc_bias is gravity only
data.qpos[:ARM] = q_ref
data.qvel[:] = 0
mujoco.mj_forward(model, data)                # computes all derived quantities without stepping time
tau_g = data.qfrc_bias[:ARM].copy()           # [N*m]

# --- Prediction
e_pred = -tau_g / (gear**2 * kp)              # [rad]

# --- Simulate the hold, 20 s at dt = 2 ms
data.ctrl[:] = ctrl_home
for _ in range(10_000):
    mujoco.mj_step(model, data)
e_meas = data.qpos[:ARM] - q_ref
print(f"residual max |qvel| = {np.abs(data.qvel[:ARM]).max():.2e} rad/s  (near 0 means steady state)")
print(f"joint damping:  {model.dof_damping[:ARM]}")
print(f"frictionloss:   {model.dof_frictionloss[:ARM]}\n")

# --- Compare
print(f"{'joint':>5} {'kp':>8} {'tau_g':>9} {'e_pred':>10} {'e_meas':>10} {'diff':>9}")
print(f"{'':>5} {'N*m/rad':>8} {'N*m':>9} {'deg':>10} {'deg':>10} {'%':>9}")
for j in range(ARM):
    p, m = np.degrees(e_pred[j]), np.degrees(e_meas[j])
    pct = f"{100 * (m - p) / p:+.1f}" if abs(p) > 0.01 else "n/a"
    print(f"{j+1:>5} {kp[j]:>8.0f} {tau_g[j]:>9.3f} {p:>10.4f} {m:>10.4f} {pct:>9}")
