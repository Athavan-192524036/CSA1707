"""
ROS 2 Launch file for Lane Detection Node
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config_path', default_value='configs/config.yaml'),
        DeclareLaunchArgument('checkpoint_path', default_value='checkpoints/best.pth'),
        DeclareLaunchArgument('input_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('device', default_value='cuda'),
        
        Node(
            package='lane_detection_agent',
            executable='lane_detection_node',
            name='lane_detection_node',
            output='screen',
            parameters=[{
                'config_path': LaunchConfiguration('config_path'),
                'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                'input_topic': LaunchConfiguration('input_topic'),
                'device': LaunchConfiguration('device'),
                'publish_rate': 30.0,
                'confidence_threshold': 0.6
            }],
            remappings=[
                ('/perception/lane_markings', '/lane_detection/lanes'),
                ('/perception/lane_visualization', '/lane_detection/visualization')
            ]
        ),
    ])