"""Tests for fk/ik_dls.py. Run from the repo root: pytest -q tests/test_ik_dls.py

Test arm: planar 2R, L1 = 0.30 m, L2 = 0.20 m (the Wk 3 Mon guided example).
"""

import numpy as np
import pytest

from fk.ik_dls import IKResult, dls_step, fd_jacobian, ik_dls

L1, L2 = 0.30, 0.20  # m


def fk_2r(theta: np.ndarray) -> np.ndarray:
    """Planar 2R tip position (x, y), m."""
    t1, t2 = theta
    return np.array(
        [
            L1 * np.cos(t1) + L2 * np.cos(t1 + t2),
            L1 * np.sin(t1) + L2 * np.sin(t1 + t2),
        ]
    )


def jac_2r(theta: np.ndarray) -> np.ndarray:
    """Analytic planar 2R Jacobian, m/rad."""
    t1, t2 = theta
    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t1 + t2), np.cos(t1 + t2)
    return np.array(
        [
            [-L1 * s1 - L2 * s12, -L2 * s12],
            [L1 * c1 + L2 * c12, L2 * c12],
        ]
    )


# ---------- fd_jacobian ----------


def test_fd_jacobian_matches_wk3_hand_values():
    theta = np.radians([30.0, 60.0])
    J = fd_jacobian(fk_2r, theta)
    expected = np.array([[-0.3500, -0.2000], [0.2598, 0.0000]])  # Wk 3 Mon, m/rad
    assert J.shape == (2, 2)
    assert np.allclose(J, expected, atol=1e-4)


def test_fd_jacobian_matches_analytic_at_random_poses():
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta = rng.uniform(-np.pi, np.pi, size=2)
        assert np.allclose(fd_jacobian(fk_2r, theta), jac_2r(theta), atol=1e-8)


def test_fd_jacobian_shape_non_square():
    # 4 joints in, 3 outputs: J must be (3, 4)
    def fk_4to3(theta):
        return np.array([theta.sum(), theta[0] * theta[1], np.sin(theta[3])])

    J = fd_jacobian(fk_4to3, np.array([0.1, 0.2, 0.3, 0.4]))
    assert J.shape == (3, 4)


def test_fd_jacobian_does_not_modify_input():
    theta = np.array([0.3, -0.2])
    before = theta.copy()
    fd_jacobian(fk_2r, theta)
    assert np.array_equal(theta, before)


# ---------- dls_step ----------


def test_dls_step_lam0_is_plain_solve():
    J = jac_2r(np.radians([30.0, 60.0]))
    err = np.array([0.010, -0.010])  # m
    assert np.allclose(dls_step(J, err, lam=0.0), np.linalg.solve(J, err), atol=1e-12)


def test_dls_step_finite_at_singularity():
    # theta2 = 0: arm fully stretched, det J = 0. lam = 0 would raise LinAlgError.
    J = jac_2r(np.array([0.0, 0.0]))
    err = np.array([0.0, 0.010])
    dtheta = dls_step(J, err, lam=0.05)
    assert dtheta.shape == (2,)
    assert np.all(np.isfinite(dtheta))


def test_dls_step_damping_shrinks_step_near_singularity():
    J = jac_2r(np.array([0.0, 0.01]))  # nearly stretched
    err = np.array([0.0, 0.010])
    small_lam = np.linalg.norm(dls_step(J, err, lam=1e-6))
    big_lam = np.linalg.norm(dls_step(J, err, lam=0.05))
    assert big_lam < small_lam


# ---------- ik_dls ----------


def test_ik_reaches_reachable_target():
    target = fk_2r(np.radians([20.0, 45.0]))
    res = ik_dls(fk_2r, np.array([0.1, 0.5]), target)
    assert isinstance(res, IKResult)
    assert res.converged
    assert res.err_norm < 1e-6
    # Check position, not angles: elbow-up and elbow-down both reach the target.
    assert np.allclose(fk_2r(res.theta), target, atol=1e-6)


def test_ik_does_not_modify_theta0():
    theta0 = np.array([0.1, 0.5])
    before = theta0.copy()
    ik_dls(fk_2r, theta0, fk_2r(np.radians([20.0, 45.0])))
    assert np.array_equal(theta0, before)


def test_ik_unreachable_target_reports_failure():
    # 0.60 m is beyond L1 + L2 = 0.50 m. Best possible error is 0.10 m.
    # lam = 0.1: with light damping the solver chatters at the stretched-arm
    # singularity instead of settling (Wed exercise).
    target = np.array([0.60, 0.0])
    res = ik_dls(fk_2r, np.array([0.2, 0.3]), target, lam=0.1, max_iters=200)
    assert not res.converged
    assert res.iters == 200
    assert res.err_norm == pytest.approx(0.10, abs=1e-3)


def test_ik_respects_joint_limits():
    # Target needs |theta2| = 45 deg; limits allow theta2 in [0, 0.5] rad only.
    target = fk_2r(np.radians([20.0, 45.0]))
    lower = np.array([-np.pi, 0.0])
    upper = np.array([np.pi, 0.5])
    res = ik_dls(fk_2r, np.array([0.1, 0.2]), target, lower=lower, upper=upper)
    assert not res.converged
    assert np.all(res.theta >= lower) and np.all(res.theta <= upper)
