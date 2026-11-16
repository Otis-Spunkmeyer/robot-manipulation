#include "so101_fk.hpp"

#include <array>
#include <cmath>

namespace so101 {

namespace {
// From assets/so101/so101_new_calib.urdf, base_link -> gripper_frame_link.
// Same values as the Python Chain; they agree with your Wk 2 S list to 4 places.
const std::array<Screw, 5> kS = [] {
    std::array<Screw, 5> s;
    s[0] << -0.0, 0.0000026536, -1.0, -0.0000001566, 0.0388353, 0.0000001031;                       // shoulder_pan
    s[1] << -0.0000063268, 1.0, 0.0000026536, -0.1166, -0.0000009214, 0.0692344329;                 // shoulder_lift
    s[2] << -0.0000063268, 1.0, 0.0000026536, -0.2291701028, -0.0000017079, 0.0972340194;           // elbow_flex
    s[3] << -0.0000063268, 1.0, 0.0000026536, -0.2343701028, -0.0000020988, 0.2321340194;           // wrist_flex
    s[4] << -0.9999999999, -0.0000099956, -0.0000065056, 0.0000023438, -0.2343684191, -0.0001796985;  // wrist_roll
    return s;
}();

const Eigen::Matrix4d kM = [] {
    Eigen::Matrix4d m;
    m << 0.000008665, -0.0000103004, 0.9999999999, 0.3913614702,
         0.0486629269, 0.9988152579, 0.0000098665, -0.0000092121,
        -0.9988152579, 0.0486629268, 0.000009156, 0.2264697102,
         0.0, 0.0, 0.0, 1.0;
    return m;
}();
}  // namespace

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    //return [w], the 3x3 skew-symmetric matrix of w.
    Eigen::Matrix3d W;
    W << 0.0, -w.z(), w.y(),
         w.z(), 0.0, -w.x(),
         -w.y(), w.x(), 0.0;
    return W;
}

Eigen::Matrix4d exp_screw(const Screw& S, double theta) {
    //Rodrigues for R, MR Eq. 3.88 for p.
    const Eigen::Vector3d w = S.head<3>();
    const Eigen::Vector3d v = S.tail<3>();
    const Eigen::Matrix3d W = skew(w);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const double s = std::sin(theta);
    const double c = std::cos(theta);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = I + s * W + (1.0 - c) * W * W;
    T.topRightCorner<3, 1>() = (I * theta + (1.0 - c) * W + (theta - s) * W * W) * v;
    return T;
}

Eigen::Matrix4d fk_space(const Vec5& theta) {
    // T = exp([S1] t1) ... exp([S5] t5) M
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 5; ++i) {
        T = T * exp_screw(kS[i], theta(i));
    }
    return T * kM;
}

}  // namespace so101
