"""Mass, COM and inertia of everything distal to a URDF joint, at q = 0.

Usage:
    python scripts/distal_props.py assets/so101/so101_new_calib.urdf wrist_flex

Everything is expressed in the joint frame (joint origin, q = 0 for this and
all downstream joints). SI units: m, kg, kg*m^2.
r_perp = distance from the joint axis. For a gravity-loaded joint, the worst
case lever arm is r_perp (COM vector perpendicular to the axis, horizontal).
"""
import sys
import xml.etree.ElementTree as ET

import numpy as np

MASSLESS = 1e-6  # kg; exporters write 1e-9 on frame-marker links


def rot_rpy(r, p, y):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def origin_T(elem):
    """4x4 transform from a URDF <origin> element (identity if missing)."""
    T = np.eye(4)
    if elem is None:
        return T
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
    T[:3, :3] = rot_rpy(*rpy)
    T[:3, 3] = xyz
    return T


def load(path):
    root = ET.parse(path).getroot()

    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            links[link.get("name")] = None
            continue
        m = float(inertial.find("mass").get("value"))
        I = inertial.find("inertia")

        def g(k):
            return float(I.get(k, "0"))

        Ic = np.array([[g("ixx"), g("ixy"), g("ixz")],
                       [g("ixy"), g("iyy"), g("iyz")],
                       [g("ixz"), g("iyz"), g("izz")]])
        links[link.get("name")] = (m, origin_T(inertial.find("origin")), Ic)

    joints = {}
    for j in root.findall("joint"):
        axis = j.find("axis")
        a = (np.array([float(v) for v in axis.get("xyz").split()])
             if axis is not None else np.array([1.0, 0.0, 0.0]))
        joints[j.get("name")] = dict(
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            T=origin_T(j.find("origin")),
            axis=a / np.linalg.norm(a) if np.linalg.norm(a) > 0 else a,  # fixed joints may export 0 0 0
        )
    return links, joints


def distal(links, joints, joint_name):
    """Walk the subtree below joint_name at q = 0.

    Returns the joint axis and a list of (link, mass, COM or frame origin,
    inertia about COM rotated into the joint frame).
    """
    if joint_name not in joints:
        sys.exit(f"joint '{joint_name}' not in URDF. Joints: {', '.join(joints)}")
    j0 = joints[joint_name]
    bodies = []
    stack = [(j0["child"], np.eye(4))]
    while stack:
        link, T = stack.pop()
        if links.get(link) is not None and links[link][0] > MASSLESS:
            m, T_in, Ic = links[link]
            Tc = T @ T_in
            R = Tc[:3, :3]
            bodies.append((link, m, Tc[:3, 3], R @ Ic @ R.T))
        else:
            bodies.append((link, 0.0, T[:3, 3], np.zeros((3, 3))))
        for j in joints.values():
            if j["parent"] == link:
                stack.append((j["child"], T @ j["T"]))
    return j0["axis"], bodies


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    path, name = sys.argv[1], sys.argv[2]
    links, joints = load(path)
    a, bodies = distal(links, joints, name)

    def perp(p):
        return np.linalg.norm(p - (p @ a) * a)

    print(f"Joint {name}: axis (joint frame) = {np.round(a, 4)}")
    print(f"{'link':<30}{'m (kg)':>9}{'x':>9}{'y':>9}{'z':>9}{'r_perp':>9}")
    m_tot, mc, I_axis = 0.0, np.zeros(3), 0.0
    for link, m, p, Ic in bodies:
        tag = f"{m:>9.4f}" if m > 0 else f"{'frame':>9}"
        print(f"{link:<30}{tag}{p[0]:>9.4f}{p[1]:>9.4f}{p[2]:>9.4f}{perp(p):>9.4f}")
        if m > 0:
            m_tot += m
            mc += m * p
            I_axis += a @ Ic @ a + m * (p @ p - (a @ p) ** 2)
    print("  (COM for links with mass; frame origin for massless links. m, joint frame, q = 0)")

    c = mc / m_tot
    print()
    print(f"m_d     = {m_tot:.4f} kg")
    print(f"COM     = {np.round(c, 4)} m")
    print(f"r_d     = {perp(c):.4f} m      (COM distance from the joint axis)")
    print(f"I_axis  = {I_axis:.4e} kg*m^2  (about the joint axis)")


if __name__ == "__main__":
    main()
