"""Wk 4: DLS damping sweep on an unreachable target (planar 2R)."""
import numpy as np

from fk.ik_dls import dls_step, fd_jacobian

L1, L2 = 0.30, 0.20  # m


def fk_2r(theta):
    t1, t2 = theta
    return np.array([L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
                     L1 * np.sin(t1) + L2 * np.sin(t1 + t2)])


target = np.array([0.60, 0.0])  # m, beyond reach (L1 + L2 = 0.50 m)

for lam in [0.01, 0.05, 0.1]:
    theta = np.array([0.2, 0.3])
    errs = []
    for _ in range(30):
        err = target - fk_2r(theta)
        errs.append(np.linalg.norm(err))
        theta = theta + dls_step(fd_jacobian(fk_2r, theta), err, lam)
    print(f"lam={lam:<5} |err| first 10 (m): " + " ".join(f"{e:.3f}" for e in errs[:10]))
    print(f"           last: {errs[-1]:.4f} m, theta2 = {np.degrees(theta[1]):.1f} deg")
