#pragma once
// SO-101 forward kinematics by product of exponentials, space form.
// Units: joint angles rad, lengths m.
#include <Eigen/Dense>

namespace so101 {

using Vec5 = Eigen::Matrix<double, 5, 1>;
using Screw = Eigen::Matrix<double, 6, 1>;  // (omega, v)

// 3x3 skew-symmetric matrix [w] such that [w] x = w cross x.
Eigen::Matrix3d skew(const Eigen::Vector3d& w);

// Matrix exponential of a revolute screw axis S (|omega| = 1) times theta.
Eigen::Matrix4d exp_screw(const Screw& S, double theta);

// T = exp([S1] t1) ... exp([S5] t5) M
Eigen::Matrix4d fk_space(const Vec5& theta);

}  // namespace so101
