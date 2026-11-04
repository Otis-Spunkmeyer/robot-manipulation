"""Week 4: position-only IK on the SO-101 chain with damped least squares."""

import sys
from pathlib import Path

import numpy as np

ROOT= Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fk.urdf_poe import build_poe # noqa:E402
from fk.ik_dls import ik_dls #noqa: E402

URDF = ROOT / "assets" / "so101" / "so101_new_calib.urdf"

def so101_position_ik(model, p_target, theta0, lam=0.1):
    """Position-only DLS IK for the SO-101 tool frame.

    Args:
        model: PoEModel, base_link -> gripper_frame_link.
        p_target: target position, shape (3,), m.
        theta0: (5,) initial joint angles, rad.
        lam: DLS damping (default 0.1), units m/rad.

    Returns:
        IKResult from ik_dls.
    """
    def fk_pos(theta):
        return model.fk_space(theta)[:3, 3]

    return ik_dls(fk_pos, theta0, p_target, lam=lam,
                   lower=model.lower, upper=model.upper)

if __name__ == "__main__":
    model = build_poe(URDF, "base_link", "gripper_frame_link")
    print("joints:", model.joint_names)
    print("lower [deg]:", np.round(np.degrees(model.lower), 1))
    print("upper [deg]:", np.round(np.degrees(model.upper), 1))

    theta_true = np.radians([30.0, 20.0, -40.0, 15.0, 0.0])
    p_target = model.fk_space(theta_true)[:3, 3]
    print("p_target [m]:", np.round(p_target, 4))
    
    theta0 = np.zeros(model.n)
    res = so101_position_ik(model, p_target, theta0)
    print(res)
    print("theta solved [deg]:", np.round(np.degrees(res.theta), 2))
    print("theta true   [deg]:", np.round(np.degrees(theta_true), 2))

    p_sol = model.fk_space(res.theta)[:3, 3]
    miss = np.linalg.norm(p_sol - p_target)
    print("FK(solved) [m]:", np.round(p_sol, 6), "|miss| [m]:", miss)

    assert res.converged, "IK did not converge"
    assert miss < 1e-6, f"miss {miss:.2e} m"
    assert model.within_limits(res.theta), "solution outside joint limits"
    print("all checks passed")
    