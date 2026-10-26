"""Broadcast a static transform world -> base_link (robot mounted on a table).

Parameters:
  parent   default "world"
  child    default "base_link"
  xyz      translation of child in parent, m       default [0.10, 0.20, 0.0]
  yaw_deg  rotation of child about parent z, deg   default 90.0
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class StaticWorld(Node):
    def __init__(self):
        super().__init__("static_world")
        self.declare_parameter("parent", "world")
        self.declare_parameter("child", "base_link")
        self.declare_parameter("xyz", [0.10, 0.20, 0.0])
        self.declare_parameter("yaw_deg", 90.0)

        xyz = list(self.get_parameter("xyz").value)
        yaw = math.radians(self.get_parameter("yaw_deg").value)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter("parent").value
        t.child_frame_id = self.get_parameter("child").value
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = xyz
        # Rotation about z by yaw: quaternion (x, y, z, w) = (0, 0, sin(yaw/2), cos(yaw/2))
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)

        # Static transforms are latched (transient_local QoS): sent once, delivered
        # to every listener, including ones that start later.
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(t)
        self.get_logger().info(
            f"{t.header.frame_id} -> {t.child_frame_id}: xyz={xyz} m, yaw={math.degrees(yaw):.1f} deg")


def main(args=None):
    rclpy.init(args=args)
    node = StaticWorld()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C: Jazzy raises ExternalShutdownException
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
