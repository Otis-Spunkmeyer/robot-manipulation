"""Damped least-squares (DLS) inverse kinematics for position targets.

Works with any forward-kinematics function that maps joint angles to a
tool position, so the same code serves the planar 2R test arm and the
SO-101 chain from fk/urdf_poe.py.

Units: joint angles in rad, positions in m.

Update rule (MR 6.2, with damping):
    e        = x_target - fk(theta)                 (m,)   m
    J        = d fk / d theta                       (m, n) m/rad
    dtheta   = J^T (J J^T + lam^2 I)^-1 e           (n,)   rad
    theta   <- theta + dtheta, clipped to joint limits

lam = 0 gives the plain Newton-Raphson / pseudoinverse step. lam > 0 keeps
the step finite at and near singularities, at the cost of slower
convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# fk(theta) -> tool position. theta shape (n,), output shape (m,), usually m = 2 or 3.
FkFn = Callable[[np.ndarray], np.ndarray]


def fd_jacobian(fk: FkFn, theta: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of fk at theta.

    Column i = (fk(theta + h e_i) - fk(theta - h e_i)) / (2 h)

    Args:
        fk: forward kinematics, theta (n,) -> position (m,).
        theta: joint angles, shape (n,), rad.
        h: step size, rad.

    Returns:
        J, shape (m, n), units m/rad.

    Notes:
        - Do not modify the caller's theta in place. Work on copies.
        - m comes from the size of fk's output, n from the size of theta.
    """
    n = theta.size
    m = fk(theta).size
    J = np.zeros((m, n))
    for i in range(n):
        dt = np.zeros(n)
        dt[i] = h
        J[:, i] = (fk(theta + dt) - fk(theta - dt)) / (2 * h)
    return J


def dls_step(J: np.ndarray, err: np.ndarray, lam: float) -> np.ndarray:
    """One damped least-squares step: dtheta = J^T (J J^T + lam^2 I)^-1 err.

    Args:
        J: Jacobian, shape (m, n), m/rad.
        err: task-space error, shape (m,), m.
        lam: damping, units of J (m/rad). lam = 0 is the undamped step.

    Returns:
        dtheta, shape (n,), rad.

    Notes:
        - Solve the (m, m) system with np.linalg.solve. Do not form an inverse.
        - I is the (m, m) identity.
    """
    m = J.shape[0]
    A = J @ J.T + lam**2 * np.eye(m)
    y = np.linalg.solve(A, err)
    return J.T @ y


@dataclass
class IKResult:
    theta: np.ndarray  # final joint angles, (n,), rad
    converged: bool  # True if |err| < tol was reached
    iters: int  # number of DLS steps taken
    err_norm: float  # final |x_target - fk(theta)|, m


def ik_dls(
    fk: FkFn,
    theta0: np.ndarray,
    target: np.ndarray,
    lam: float = 0.01,
    tol: float = 1e-6,
    max_iters: int = 100,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> IKResult:
    """Iterate DLS steps until the position error is below tol.

    Loop, at most max_iters times:
        1. err = target - fk(theta)
        2. if |err| < tol: stop, converged
        3. J = fd_jacobian(fk, theta)
        4. theta = theta + dls_step(J, err, lam)
        5. if limits were given: clip theta to [lower, upper]

    After the loop, compute the final error once more for err_norm.

    Args:
        fk: forward kinematics, theta (n,) -> position (m,).
        theta0: initial guess, (n,), rad. Do not modify it.
        target: desired position, (m,), m.
        lam: damping, m/rad.
        tol: convergence threshold on |err|, m.
        max_iters: maximum number of steps.
        lower, upper: joint limits, (n,), rad. Both or neither.

    Returns:
        IKResult.
    """
    theta = np.array(theta0, dtype=float)  # work on a copy
    for k in range(max_iters):
        err = target - fk(theta)
        if np.linalg.norm(err) < tol:
            return IKResult(theta, True, k, float(np.linalg.norm(err)))
        J = fd_jacobian(fk, theta)
        theta = theta +dls_step(J, err, lam)
        if lower is not None:
            theta = np.clip(theta, lower, upper)
    err_norm = float(np.linalg.norm(target-fk(theta)))
    return IKResult(theta, err_norm < tol, max_iters, err_norm)
    
