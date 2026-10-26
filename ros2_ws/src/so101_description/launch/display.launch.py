"""SO-101 in RViz.

Args:
  gui:=true|false        start joint_state_publisher_gui (false = drive /joint_states yourself)
  world:=true|false      start the static world -> base_link broadcaster
  rviz_config:=<path>    RViz config (default: share/so101_description/rviz/so101.rviz)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("so101_description")
    with open(os.path.join(share, "urdf", "so101.urdf")) as f:
        robot_description = f.read()

    gui = LaunchConfiguration("gui")
    world = LaunchConfiguration("world")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("world", default_value="false"),
        DeclareLaunchArgument(
            "rviz_config", default_value=os.path.join(share, "rviz", "so101.rviz")),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            condition=IfCondition(gui),
        ),
        Node(
            package="so101_nodes_py",
            executable="static_world",
            condition=IfCondition(world),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_config],
        ),
    ])
