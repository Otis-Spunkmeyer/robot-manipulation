"""One Newton-Raphson IK step on SO-101 J2-J3 chain.

Units: m, rad. Planar chain in the x-z plane, both axes +y."""

import numpy as np

# --- 1. Data (from Wk 3 P1) --- 
q2 = np.array([0.0692, 0.0, 0.1166]) # m, J2 axis point
q3 = np.array([0.0972, 0.0, 0.2292]) # m, J3 axis point
p0 = np.array([0.2321, 0.0, 0.2344]) # m, J4 axis point at home
w_hat = np.array([0.0, 1.0, 0.0])  # unit vector rotation axis

# --- 2. Rotation about +y ---
def Ry(t):
    """Rotation matrix about +y and t rad."""
    c,s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

# sanity checks
assert np.allclose(Ry(0.0), np.eye(3))
assert np.isclose(np.linalg.det(Ry(0.7)), 1.0)
assert np.allclose(Ry(np.pi / 2) @ np.array([1, 0, 0]), [0, 0, -1])
print("Ry checks passed")

# --- 3. Forward kinematics ---
def fk(t2, t3):
    """Position of the J4 axis point for joint angles t2, t3 (rad). Returns m."""
    return q2 +Ry(t2) @ ((q3 - q2) + Ry(t3) @ (p0-q3))

# checks
assert np.allclose(fk(0.0, 0.0), p0)
r_home = np.linalg.norm(p0 - q2)
assert np.isclose(np.linalg.norm(fk(0.3, 0.0) -q2), r_home)
print("fk(0,0) =" , fk(0.0, 0.0))

# --- 4. Jacobian, two ways ---

def jac_analytic(t2, t3):
    """2x2 Jacobian (rows x, z), m/rad, from omega x (p-q)."""
    p = fk(t2, t3)
    q3_now = q2 +Ry(t2) @ (q3-q2)   # elbow axis moves when the shoulder turns
    c2 = np.cross(w_hat, p-q2)
    c3 = np.cross(w_hat, p-q3_now)
    return np.array([[c2[0], c3[0]], 
                    [c2[2], c3[2]]])

def jac_fd(t2, t3, h=1e-6):
    """2x2 jacobioan by central finite differences, m/rad."""
    J = np.zeros((2, 2))
    t = np.array([t2, t3])
    for i in range(2):
        dt = np.zeros(2)
        dt[i] = h
        dp = fk(*(t + dt)) - fk(*(t -dt))
        J[:, i] = dp[[0, 2]] / (2*h)
    return J

J = jac_analytic(0.0, 0.0)
print("J analytic =\n", J)
print("max |analytic -fd| =", np.max(np.abs(J - jac_fd(0.0, 0.0))))
assert np.allclose(J, jac_fd(0.0, 0.0), atol=1e-8)
assert np.allclose(jac_analytic(0.4, -0.6), jac_fd(0.4, -0.6), atol=1e-8)

# --- 5. One Newton-Raphson step ---
dp_target = np.array([0.010, -0.010])   # m, (x, z): 10mm out, 10mm down
dtheta = np.linalg.solve(J, dp_target)  # rad
print("dtheta (rad) =", dtheta, " (deg) =", np.degrees(dtheta))
assert np.allclose(dtheta, [0.08621, -0.02998], atol=1e-5) # hand calculation result

p1 = fk(*dtheta)    # *dtheta unpacks the two angles into fk
achieved = (p1-p0)[[0, 2]]              # m, (x, z)
risidual = dp_target - achieved         # m
r_mm = np.linalg.norm(risidual) * 1000.0
pct = 100 * np.linalg.norm(risidual) / np.linalg.norm(dp_target)
print("achieved dp (mm) =", achieved * 1e3)
print("residual (mm)    =", risidual * 1e3)
print(f"|residual| = {r_mm:.3f} mm = {pct:.2f}% of the step")

# --- 6. Iterate to convergence ---
target = p0[[0, 2]] + dp_target     # m, (x, z): where we want the J4 point
t = np.zeros(2)
for k in range(6):
    err = target - fk(*t)[[0, 2]]
    print(f"iter {k}: |err| = {np.linalg.norm(err):.3e} m")
    if np.linalg.norm(err) < 1e-9:      # 1 nm: done; np.linalg.norm(err) is the Euclidean norm of the 2D error vector
        break
    t = t + np.linalg.solve(jac_analytic(*t), err)   # Newton step from the CURRENT pose
print("converged theta (deg) =", np.degrees(t))
assert np.allclose(fk(*t)[[0, 2]], target, atol=1e-9)