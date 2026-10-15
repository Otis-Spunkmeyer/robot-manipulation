"""Week 1 lab, part 3: Coulomb friction (frictionloss) and viscous damping vs P-servo hold error."""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path.home() / "dev" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
ARM = 7


def hold(frictionloss=0.0, damping=1.0, start_offset_deg=None, steps=10_000):
    """Hold the home pose for steps*dt seconds. Returns (final error per joint [deg], j2 settling time [s], model)."""
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.dof_frictionloss[:ARM] = frictionloss   # Coulomb friction limit per joint [N*m]
    model.dof_damping[:ARM] = damping             # viscous damping per joint [N*m*s/rad]
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    ctrl = model.key_ctrl[0].copy()
    q_ref = ctrl[:ARM].copy()                     # gear = 1, so commanded position = joint reference
    if start_offset_deg is not None:              # start away from q_ref to test approach direction
        data.qpos[:ARM] = q_ref + np.radians(start_offset_deg)
    data.ctrl[:] = ctrl

    hist = np.empty((steps, ARM))
    for k in range(steps):
        mujoco.mj_step(model, data)
        hist[k] = data.qpos[:ARM] - q_ref
    e = np.degrees(hist)
    e_final = e[-1]

    # j2 settling time: last instant its error was outside a 2% band around the final value
    j = 1
    band = 0.02 * abs(e_final[j])
    outside = np.where(np.abs(e[:, j] - e_final[j]) > band)[0]
    t_settle = (outside[-1] + 1) * model.opt.timestep if outside.size else 0.0
    return e_final, t_settle, model


TF = 2.0  # Coulomb friction to test [N*m]

e0, ts0, model = hold()                                       # baseline: frictionless, damping 1
e_near, _, _ = hold(frictionloss=TF)                          # friction, start at q_ref
e_far, _, _ = hold(frictionloss=TF, start_offset_deg=2 * e0)  # friction, start beyond the sag
e_d, tsd, _ = hold(damping=20.0)                              # frictionless, damping 20

kv = -model.actuator_biasprm[:ARM, 2]                         # servo derivative gain [N*m*s/rad]
print(f"servo kv (j1..j7): {kv}\n")
print(f"{'case':<40} {'j2 e [deg]':>11} {'j4 e [deg]':>11} {'j2 settle [s]':>14}")
print(f"{'frictionless, damping 1':<40} {e0[1]:>11.4f} {e0[3]:>11.4f} {ts0:>14.3f}")
print(f"{f'friction {TF} N*m, start at q_ref':<40} {e_near[1]:>11.4f} {e_near[3]:>11.4f} {'':>14}")
print(f"{f'friction {TF} N*m, start beyond sag':<40} {e_far[1]:>11.4f} {e_far[3]:>11.4f} {'':>14}")
print(f"{'frictionless, damping 20':<40} {e_d[1]:>11.4f} {e_d[3]:>11.4f} {tsd:>14.3f}")
