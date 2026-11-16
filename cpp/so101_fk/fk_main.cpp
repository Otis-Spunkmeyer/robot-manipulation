#include <cmath>
#include <iomanip>
#include <iostream>

#include "so101_fk.hpp"

int main() {
    const double deg = M_PI / 180.0;
    so101::Vec5 theta;
    theta << 30.0 * deg, 20.0 * deg, -40.0 * deg, 15.0 * deg, 0.0 * deg;

    const Eigen::Matrix4d T = so101::fk_space(theta);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "T =\n" << T << "\n";
    std::cout << "p = " << T.topRightCorner<3, 1>().transpose() << " m\n";
    return 0;
}
