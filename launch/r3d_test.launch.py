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
            remappings=[('/output_depth', '/iphone/rgbd'),
                        ('/output_depth', '/iphone/cloud'),
                        ('/output_depth', '/iphone/confidence'),
                        ('/output_depth', '/iphone/pose'),
                        ('/output_depth', '/iphone/cloud'),],
            parameters=[config]
        ),
    ])
