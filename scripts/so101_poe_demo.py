"""Print the SO-101 PoE model built from the URDF and one FK evaluation.

Run from the repo root:  python scripts/so101_poe_demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fk.urdf_poe import build_poe  # noqa: E402

URDF = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101_new_calib.urdf"

np.set_printoptions(precision=4, suppress=True)
model = build_poe(URDF, "base_link", "gripper_frame_link")

print("Joints:", model.joint_names)
print("\nM (tool frame at home, in base frame):\n", model.M)
print("\nS list (rows = joints; w | v, v in m):\n", model.Slist.T)
print("\nB list (rows = joints; w | v, v in m):\n", model.Blist.T)

theta = np.deg2rad([30, 20, -40, 15, 0])
T = model.fk_space(theta)
print("\ntheta (deg):", np.rad2deg(theta))
print("T_space:\n", T)
print("tool point (m):", T[:3, 3])
print("space == body:", np.allclose(T, model.fk_body(theta)))
print("space == URDF chain:", np.allclose(T, model.fk_urdf(theta)))
