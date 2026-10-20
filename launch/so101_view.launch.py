"""View the SO-101 URDF in RViz. gui:=false to drive /joint_states yourself."""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

URDF = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101_rviz.urdf"


def generate_launch_description():
    gui = LaunchConfiguration("gui")
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true",
                              description="Start joint_state_publisher_gui"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": URDF.read_text()}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            condition=IfCondition(gui),
        ),
        Node(package="rviz2", executable="rviz2"),
    ])
