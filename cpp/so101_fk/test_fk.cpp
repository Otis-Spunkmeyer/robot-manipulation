// Minimal unit test: returns 0 on pass, 1 on fail. CTest reads the exit code.
#include <cmath>
#include <iostream>

#include "so101_fk.hpp"

int failures = 0;

void check_near(const char* what, double got, double want, double tol) {
    const bool ok = std::abs(got - want) <= tol;
    std::cout << (ok ? "PASS " : "FAIL ") << what << ": got " << got << ", want " << want << "\n";
    if (!ok) ++failures;
}

int main() {
    const double deg = M_PI / 180.0;

    // 1. Home pose: T = M, tool at (0.3914, 0, 0.2265) m.
    {
        const Eigen::Matrix4d T = so101::fk_space(so101::Vec5::Zero());
        check_near("home x", T(0, 3), 0.391361, 1e-6);
        check_near("home y", T(1, 3), -0.000009, 1e-6);
        check_near("home z", T(2, 3), 0.226470, 1e-6);
    }

    // 2. Wk 2 pose (30, 20, -40, 15, 0) deg vs Python fk_space and tf2_echo.
    {
        so101::Vec5 th;
        th << 30 * deg, 20 * deg, -40 * deg, 15 * deg, 0 * deg;
        const Eigen::Matrix4d T = so101::fk_space(th);
        check_near("wk2 x", T(0, 3), 0.367494, 1e-6);
        check_near("wk2 y", T(1, 3), -0.189762, 1e-6);
        check_near("wk2 z", T(2, 3), 0.269837, 1e-6);
    }

    // 3. Rotation stays a rotation: R^T R = I and det R = +1.
    {
        so101::Vec5 th;
        th << 10 * deg, -30 * deg, 45 * deg, -20 * deg, 60 * deg;
        const Eigen::Matrix3d R = so101::fk_space(th).topLeftCorner<3, 3>();
        check_near("orthonormal", (R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-9);
        check_near("det R", R.determinant(), 1.0, 1e-9);
    }

    // 4. Every joint moving, including wrist roll (S5), vs Python fk_space.
    {
        so101::Vec5 th;
        th << 10 * deg, -30 * deg, 45 * deg, -20 * deg, 60 * deg;
        const Eigen::Matrix4d T = so101::fk_space(th);
        check_near("all-joint x", T(0, 3), 0.322231, 1e-6);
        check_near("all-joint y", T(1, 3), -0.057013, 1e-6);
        check_near("all-joint z", T(2, 3), 0.207995, 1e-6);
    }

    std::cout << (failures == 0 ? "ALL PASS\n" : "FAILURES\n");
    return failures == 0 ? 0 : 1;
}
