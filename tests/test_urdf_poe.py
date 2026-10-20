"""Tests for fk/urdf_poe.py against the SO-101 URDF and Wk 2 Mon hand calcs."""

from pathlib import Path

import numpy as np
import pytest

from fk.urdf_poe import build_poe, rpy_to_R

URDF = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101_new_calib.urdf"
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@pytest.fixture(scope="module")
def arm():
    """Full 5-DOF arm to the gripper tool frame."""
    return build_poe(URDF, "base_link", "gripper_frame_link")


@pytest.fixture(scope="module")
def to_wrist():
    """Tip at the wrist_flex joint (Wk 2 Mon Problem 2).

    wrist_link is the child of wrist_flex, so the chain has 4 joints. Rotating
    wrist_flex does not move its own origin, so theta4 = 0 reproduces Problem 2.
    """
    return build_poe(URDF, "base_link", "wrist_link")


@pytest.fixture(scope="module")
def to_roll():
    """J1-4, tip at the wrist_roll joint (Wk 2 Mon Problem 3)."""
    return build_poe(URDF, "base_link", "gripper_link")


def random_configs(model, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(model.lower, model.upper, size=(n, model.n))


# --- structure -------------------------------------------------------------

def test_joint_order(arm):
    assert arm.joint_names == ARM


def test_rpy_is_rotation():
    R = rpy_to_R(0.3, -1.1, 2.0)
    assert np.allclose(R.T @ R, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_screw_axes_well_formed(arm):
    for S in arm.Slist.T:
        w, v = S[:3], S[3:]
        assert np.isclose(np.linalg.norm(w), 1.0)   # revolute: unit omega
        assert np.isclose(w @ v, 0.0, atol=1e-12)   # pure rotation: v perpendicular to omega


def test_home_is_M(arm):
    assert np.allclose(arm.fk_space(np.zeros(arm.n)), arm.M)


# --- independent cross-checks ---------------------------------------------

def test_space_matches_urdf_chain(arm):
    for th in random_configs(arm):
        assert np.allclose(arm.fk_space(th), arm.fk_urdf(th), atol=1e-12)


def test_space_matches_body(arm):
    for th in random_configs(arm, seed=1):
        assert np.allclose(arm.fk_space(th), arm.fk_body(th), atol=1e-12)


# --- Monday hand calcs (rounded to 0.1 mm in the problem table) -------------

def test_monday_S_list(to_roll):
    expected = np.array([
        [0, 0, -1,  0.0,     0.0388, 0.0],
        [0, 1,  0, -0.1166,  0.0,    0.0692],
        [0, 1,  0, -0.2292,  0.0,    0.0972],
        [0, 1,  0, -0.2344,  0.0,    0.2321],
    ]).T
    # gripper_link sits below wrist_roll, so the chain has 5 joints; Monday covered J1-4.
    assert np.allclose(to_roll.Slist[:, :4], expected, atol=1e-4)


def test_monday_problem2_wrist_point(to_wrist):
    th = np.deg2rad([30, 20, -40, 0])
    p = to_wrist.fk_space(th)[:3, 3]
    assert np.allclose(p, [0.2204, -0.1260, 0.2639], atol=2e-4)


def test_monday_problem3_roll_point(to_roll):
    assert np.allclose(to_roll.M[:3, 3], [0.2932, -0.0002, 0.2344], atol=1e-4)


def test_joint_directions(to_wrist):
    """+5 deg on each joint moves the wrist point the way Monday's (b) found."""
    d = np.deg2rad(5)
    p0 = to_wrist.M[:3, 3]
    dp1 = to_wrist.fk_space([d, 0, 0, 0])[:3, 3] - p0
    dp2 = to_wrist.fk_space([0, d, 0, 0])[:3, 3] - p0
    dp3 = to_wrist.fk_space([0, 0, d, 0])[:3, 3] - p0
    assert dp1[1] < 0          # pan swings toward -y
    assert dp2[2] < 0          # shoulder_lift tips the arm down
    assert dp3[2] < 0          # elbow_flex drops the forearm


# --- input validation and joint limits (build extension) -------------------

def test_accepts_list_and_tuple(arm):
    th = [0.1, -0.2, 0.3, 0.0, 0.5]
    assert np.allclose(arm.fk_space(th), arm.fk_space(tuple(th)))
    assert np.allclose(arm.fk_space(th), arm.fk_space(np.array(th)))


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros(6), np.zeros((5, 1)), 0.0])
def test_wrong_shape_raises(arm, bad):
    with pytest.raises(ValueError, match="shape"):
        arm.fk_space(bad)


@pytest.mark.parametrize("val", [np.nan, np.inf, -np.inf])
def test_nonfinite_raises_and_names_joint(arm, val):
    th = np.zeros(5)
    th[2] = val
    with pytest.raises(ValueError, match="elbow_flex"):
        arm.fk_space(th)


def test_non_numeric_raises(arm):
    with pytest.raises(ValueError, match="numeric"):
        arm.fk_space(["a", 0, 0, 0, 0])


def test_limits_parsed_from_urdf(arm):
    # shoulder_pan: +/-1.91986 rad (110 deg) in so101_new_calib.urdf
    assert np.isclose(arm.lower[0], -1.91986)
    assert np.isclose(arm.upper[0], 1.91986)
    assert np.all(arm.lower < arm.upper)


def test_within_limits_edges(arm):
    assert arm.within_limits(np.zeros(5))
    assert arm.within_limits(arm.upper)                 # boundary is inside
    over = arm.upper.copy()
    over[1] += 1e-3
    assert not arm.within_limits(over)
    assert arm.within_limits(over, tol=2e-3)            # tolerance band


def test_limit_violations_reports_only_bad_joints(arm):
    th = np.zeros(5)
    th[0] = arm.upper[0] + 0.1
    th[3] = arm.lower[3] - 0.1
    bad = arm.limit_violations(th)
    assert set(bad) == {"shoulder_pan", "wrist_flex"}
    v, lo, hi = bad["shoulder_pan"]
    assert v > hi


def test_check_limits_flag(arm):
    th = np.zeros(5)
    th[0] = 2.5                                         # 143 deg, past 110 deg
    arm.fk_space(th)                                    # default: no check
    with pytest.raises(ValueError, match="shoulder_pan"):
        arm.fk_space(th, check_limits=True)
    with pytest.raises(ValueError, match="shoulder_pan"):
        arm.fk_body(th, check_limits=True)


def test_clip_to_limits(arm):
    th = np.array([5.0, -5.0, 0.1, 0.0, 0.0])
    c = arm.clip_to_limits(th)
    assert arm.within_limits(c)
    assert np.isclose(c[0], arm.upper[0]) and np.isclose(c[1], arm.lower[1])
    assert np.allclose(c[2:], th[2:])                   # in-range joints untouched
