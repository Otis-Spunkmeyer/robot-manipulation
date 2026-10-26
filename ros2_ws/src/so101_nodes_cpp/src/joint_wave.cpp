// Publish sensor_msgs/JointState for the SO-101. C++ port of so101_nodes_py/joint_wave.py.
//
// Parameters: rate_hz (startup only), mode ("wave" | "hold"), amplitude_deg, period_s,
// hold_deg (6 values, deg). Same names and defaults as the Python node.

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace
{
const std::vector<std::string> kJoints = {
  "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"};
constexpr double kDegToRad = M_PI / 180.0;
}  // namespace

class JointWave : public rclcpp::Node
{
public:
  JointWave()
  : Node("joint_wave_cpp")
  {
    const double rate_hz = declare_parameter<double>("rate_hz", 50.0);
    declare_parameter<std::string>("mode", "wave");
    declare_parameter<double>("amplitude_deg", 20.0);
    declare_parameter<double>("period_s", 4.0);
    declare_parameter<std::vector<double>>("hold_deg", {30.0, 20.0, -40.0, 15.0, 0.0, 0.0});

    pub_ = create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    t0_ = now();

    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / rate_hz));
    timer_ = create_wall_timer(period, [this]() {tick();});
    RCLCPP_INFO(get_logger(), "publishing /joint_states at %.1f Hz", rate_hz);
  }

private:
  bool positions(double t, std::vector<double> & pos)
  {
    pos.assign(kJoints.size(), 0.0);
    if (get_parameter("mode").as_string() == "hold") {
      const auto hold = get_parameter("hold_deg").as_double_array();
      if (hold.size() != kJoints.size()) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "hold_deg needs %zu values, got %zu", kJoints.size(), hold.size());
        return false;
      }
      for (size_t i = 0; i < hold.size(); ++i) {
        pos[i] = hold[i] * kDegToRad;
      }
      return true;
    }
    const double amp = get_parameter("amplitude_deg").as_double() * kDegToRad;
    const double w = 2.0 * M_PI / get_parameter("period_s").as_double();
    // Arm joints phase-shifted by 60 deg each; gripper (index 5) held at 0.
    for (size_t i = 0; i < 5; ++i) {
      pos[i] = amp * std::sin(w * t + static_cast<double>(i) * M_PI / 3.0);
    }
    return true;
  }

  void tick()
  {
    const rclcpp::Time stamp = now();
    std::vector<double> pos;
    if (!positions((stamp - t0_).seconds(), pos)) {
      return;
    }
    sensor_msgs::msg::JointState msg;
    msg.header.stamp = stamp;
    msg.name = kJoints;
    msg.position = pos;
    pub_->publish(msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time t0_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JointWave>());
  rclcpp::shutdown();
  return 0;
}
