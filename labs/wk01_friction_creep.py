"""Week 1 lab, part 3b: does MuJoCo frictionloss hold a dead band, or creep? Error vs time at two friction stiffnesses."""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path.home() / "dev" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
ARM, J2 = 7, 1
TF = 2.0                                     # Coulomb friction limit [N*m]
CHECK_S = [0.25, 1.0, 5.0, 20.0]             # times to sample the j2 error [s]


def run(solref_tc, start_offset_deg):
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.dof_frictionloss[:ARM] = TF
    model.dof_solref[:ARM, 0] = solref_tc    # friction constraint time constant [s]; smaller = stiffer
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    ctrl = model.key_ctrl[0].copy()
    q_ref = ctrl[:ARM].copy()
    data.qpos[J2] = q_ref[J2] + np.radians(start_offset_deg)
    data.ctrl[:] = ctrl
    dt = model.opt.timestep
    checks = {int(round(t / dt)): t for t in CHECK_S}
    out = []
    for k in range(1, max(checks) + 1):
        mujoco.mj_step(model, data)
        if k in checks:
            out.append(np.degrees(data.qpos[J2] - q_ref[J2]))
    return out


print(f"j2 error [deg] at t = {CHECK_S} s, friction {TF} N*m\n")
for tc in (0.02, 0.004):                     # 0.02 = MuJoCo default; 0.004 = stiffest allowed at dt = 2 ms
    near = run(tc, 0.0)                      # start at q_ref, sag downward
    far = run(tc, 2 * 0.3771)                # start beyond the sag, spring back
    print(f"solref time const {tc:>6} s  near: " + "  ".join(f"{v:.4f}" for v in near))
    print(f"{'':>24}  far:  " + "  ".join(f"{v:.4f}" for v in far))
