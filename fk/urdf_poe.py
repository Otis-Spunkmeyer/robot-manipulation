"""Forward kinematics for a serial URDF chain using the product of exponentials.

Reads a URDF, walks the joint chain from a base link to a tip link, and builds
the PoE ingredients (M, space screw axes S, body screw axes B) from the zero pose.
Also provides a direct URDF chain FK (compose joint origins and joint rotations)
as an independent reference for testing.

Conventions follow Lynch & Park, Modern Robotics, Ch. 4:
    T(theta) = e^[S1]th1 ... e^[Sn]thn M        (space form)
    T(theta) = M e^[B1]th1 ... e^[Bn]thn        (body form)
Units: meters, radians.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import modern_robotics as mr
import numpy as np


# ---------------------------------------------------------------------------
# Small SE(3) helpers
# ---------------------------------------------------------------------------

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (fixed axes x, y, z)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def axis_angle_T(axis: np.ndarray, theta: float) -> np.ndarray:
    """Pure rotation about a unit axis through the origin (Rodrigues via MR)."""
    return make_T(mr.MatrixExp3(mr.VecToso3(axis) * theta), np.zeros(3))


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

@dataclass
class Joint:
    name: str
    type: str                # "revolute", "continuous", "prismatic", "fixed"
    parent: str
    child: str
    T_origin: np.ndarray     # parent link frame -> joint frame at zero pose
    axis: np.ndarray         # unit axis in the joint frame
    lower: float | None = None
    upper: float | None = None


def _parse_floats(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def parse_joints(urdf_path: str | Path) -> dict[str, Joint]:
    """Return every joint in the URDF keyed by child link name."""
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, Joint] = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = _parse_floats(origin.get("xyz") if origin is not None else None, (0, 0, 0))
        rpy = _parse_floats(origin.get("rpy") if origin is not None else None, (0, 0, 0))
        axis_el = j.find("axis")
        axis = _parse_floats(axis_el.get("xyz") if axis_el is not None else None, (1, 0, 0))
        norm = np.linalg.norm(axis)
        if norm > 0:                      # fixed joints may carry a zero axis
            axis = axis / norm
        limit = j.find("limit")
        lower = float(limit.get("lower")) if limit is not None and limit.get("lower") else None
        upper = float(limit.get("upper")) if limit is not None and limit.get("upper") else None
        child = j.find("child").get("link")
        joints[child] = Joint(
            name=j.get("name"),
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=child,
            T_origin=make_T(rpy_to_R(*rpy), xyz),
            axis=axis,
            lower=lower,
            upper=upper,
        )
    return joints


def chain_between(joints: dict[str, Joint], base: str, tip: str) -> list[Joint]:
    """Walk child -> parent from tip back to base; return joints in base-to-tip order."""
    chain: list[Joint] = []
    link = tip
    while link != base:
        if link not in joints:
            raise ValueError(f"No joint has child link '{link}'; '{tip}' is not below '{base}'.")
        j = joints[link]
        chain.append(j)
        link = j.parent
    return chain[::-1]


# ---------------------------------------------------------------------------
# PoE model
# ---------------------------------------------------------------------------

MOVABLE = {"revolute", "continuous", "prismatic"}


@dataclass
class PoEModel:
    joint_names: list[str]
    M: np.ndarray            # 4x4 home pose of the tip frame in the base frame
    Slist: np.ndarray        # 6 x n space screw axes (columns)
    Blist: np.ndarray        # 6 x n body screw axes (columns)
    lower: np.ndarray        # joint lower limits (rad or m), nan if absent
    upper: np.ndarray
    chain: list[Joint] = field(repr=False)

    @property
    def n(self) -> int:
        return len(self.joint_names)

    # --- input validation and limits ---------------------------------------

    def validate(self, theta) -> np.ndarray:
        """Return theta as a float array of shape (n,); raise ValueError if malformed."""
        try:
            th = np.asarray(theta, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"theta must be numeric, got {theta!r}") from e
        if th.shape != (self.n,):
            raise ValueError(
                f"theta must have shape ({self.n},) for joints {self.joint_names}, got {th.shape}"
            )
        if not np.all(np.isfinite(th)):
            bad = [self.joint_names[i] for i in np.flatnonzero(~np.isfinite(th))]
            raise ValueError(f"theta has NaN/inf for joints {bad}")
        return th

    def limit_violations(self, theta, tol: float = 0.0) -> dict[str, tuple[float, float, float]]:
        """Joints outside [lower - tol, upper + tol]: name -> (value, lower, upper).

        Joints with no limit in the URDF (nan) are never reported.
        """
        th = self.validate(theta)
        out = {}
        for name, v, lo, hi in zip(self.joint_names, th, self.lower, self.upper):
            if (not np.isnan(lo) and v < lo - tol) or (not np.isnan(hi) and v > hi + tol):
                out[name] = (float(v), float(lo), float(hi))
        return out

    def within_limits(self, theta, tol: float = 0.0) -> bool:
        return not self.limit_violations(theta, tol)

    def clip_to_limits(self, theta) -> np.ndarray:
        """Clamp each joint into its URDF range (unlimited joints pass through)."""
        th = self.validate(theta)
        lo = np.where(np.isnan(self.lower), -np.inf, self.lower)
        hi = np.where(np.isnan(self.upper), np.inf, self.upper)
        return np.clip(th, lo, hi)

    def _prep(self, theta, check_limits: bool) -> np.ndarray:
        th = self.validate(theta)
        if check_limits:
            bad = self.limit_violations(th)
            if bad:
                detail = ", ".join(
                    f"{k}={np.rad2deg(v):.1f} deg (limits {np.rad2deg(lo):.1f} to {np.rad2deg(hi):.1f})"
                    for k, (v, lo, hi) in bad.items()
                )
                raise ValueError(f"joint limits exceeded: {detail}")
        return th

    # --- forward kinematics -------------------------------------------------

    def fk_space(self, theta, check_limits: bool = False) -> np.ndarray:
        return mr.FKinSpace(self.M, self.Slist, self._prep(theta, check_limits))

    def fk_body(self, theta, check_limits: bool = False) -> np.ndarray:
        return mr.FKinBody(self.M, self.Blist, self._prep(theta, check_limits))

    def fk_urdf(self, theta) -> np.ndarray:
        """Reference FK: compose URDF joint origins and joint motions directly.

        Independent of the screw axes, so it catches errors in building S and B.
        """
        theta = self.validate(theta)
        T = np.eye(4)
        k = 0
        for j in self.chain:
            T = T @ j.T_origin
            if j.type in ("revolute", "continuous"):
                T = T @ axis_angle_T(j.axis, theta[k])
                k += 1
            elif j.type == "prismatic":
                T = T @ make_T(np.eye(3), j.axis * theta[k])
                k += 1
        return T


def build_poe(urdf_path: str | Path, base: str, tip: str) -> PoEModel:
    """Build M, S and B for the chain base -> tip at the URDF zero pose.

    For each movable joint at home:
        omega_s = R_sj @ axis_j          (axis direction in {s})
        q_s     = p_sj                   (a point on the axis: the joint frame origin)
        revolute:  S = (omega_s, -omega_s x q_s)
        prismatic: S = (0, omega_s)
    B = [Ad_{M^-1}] S.
    """
    chain = chain_between(parse_joints(urdf_path), base, tip)

    T_home = np.eye(4)
    names, S_cols, lo, hi = [], [], [], []
    for j in chain:
        T_home = T_home @ j.T_origin          # joint frame at home, in {s}
        if j.type not in MOVABLE:
            continue
        R, p = T_home[:3, :3], T_home[:3, 3]
        w = R @ j.axis
        if j.type == "prismatic":
            S = np.r_[np.zeros(3), w]
        else:
            S = np.r_[w, -np.cross(w, p)]
        names.append(j.name)
        S_cols.append(S)
        lo.append(np.nan if j.lower is None else j.lower)
        hi.append(np.nan if j.upper is None else j.upper)

    M = T_home
    Slist = np.array(S_cols).T
    Blist = mr.Adjoint(mr.TransInv(M)) @ Slist
    return PoEModel(names, M, Slist, Blist, np.array(lo), np.array(hi), chain)
