"""Publish sensor_msgs/JointState for the SO-101.

Parameters (all can be changed live with `ros2 param set`):
  rate_hz        publish rate, Hz (read at startup)           default 50.0
  mode           "wave" (sinusoid per joint) or "hold"         default "wave"
  amplitude_deg  wave amplitude, deg                           default 20.0
  period_s       wave period, s                                default 4.0
  hold_deg       6 joint angles for "hold", deg                default [30, 20, -40, 15, 0, 0]
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class JointWave(Node):
    def __init__(self):
        super().__init__("joint_wave")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("mode", "wave")
        self.declare_parameter("amplitude_deg", 20.0)
        self.declare_parameter("period_s", 4.0)
        self.declare_parameter("hold_deg", [30.0, 20.0, -40.0, 15.0, 0.0, 0.0])

        rate_hz = self.get_parameter("rate_hz").value
        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.t0 = self.get_clock().now()
        self.timer = self.create_timer(1.0 / rate_hz, self.tick)
        self.get_logger().info(f"publishing /joint_states at {rate_hz:.1f} Hz")

    def positions(self, t: float) -> list[float] | None:
        mode = self.get_parameter("mode").value
        if mode == "hold":
            hold = list(self.get_parameter("hold_deg").value)
            if len(hold) != len(JOINTS):
                self.get_logger().error(
                    f"hold_deg needs {len(JOINTS)} values, got {len(hold)}",
                    throttle_duration_sec=2.0)
                return None
            return [math.radians(d) for d in hold]

        amp = math.radians(self.get_parameter("amplitude_deg").value)
        w = 2.0 * math.pi / self.get_parameter("period_s").value
        # Arm joints phase-shifted by 60 deg each; gripper held at 0.
        return [amp * math.sin(w * t + i * math.pi / 3.0) for i in range(5)] + [0.0]

    def tick(self):
        now = self.get_clock().now()
        pos = self.positions((now - self.t0).nanoseconds * 1e-9)
        if pos is None:
            return
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = JOINTS
        msg.position = pos
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointWave()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C: Jazzy raises ExternalShutdownException
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
