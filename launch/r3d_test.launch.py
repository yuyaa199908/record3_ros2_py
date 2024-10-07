import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('record3d_ros2_py'),
        'config',
        'param.yaml'
    )
    return LaunchDescription([
        Node(
            package='record3d_ros2_py',
            namespace='record3d_camera_node',
            executable='record3d_camera_node',
            remappings=[('/output_depth', '/iphone/depth'),
                        ('/output_color', '/iphone/color'),
                        ('/output_confidence', '/iphone/confidence'),
                        ('/output_pose', '/iphone/pose'),
                        ('/output_cloud', '/iphone/cloud'),],
            parameters=[config]
        ),
    ])
