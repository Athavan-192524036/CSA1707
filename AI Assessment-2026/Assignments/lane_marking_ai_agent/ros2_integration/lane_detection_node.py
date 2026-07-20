"""
ROS 2 Integration Node for Lane Detection
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header, Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from scripts.inference import LaneDetector


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        
        self.declare_parameter('config_path', 'configs/config.yaml')
        self.declare_parameter('checkpoint_path', 'checkpoints/best.pth')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/perception/lane_markings')
        self.declare_parameter('visualization_topic', '/perception/lane_visualization')
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('confidence_threshold', 0.6)
        
        config_path = self.get_parameter('config_path').value
        checkpoint_path = self.get_parameter('checkpoint_path').value
        
        self.get_logger().info('Loading lane detection model...')
        self.detector = LaneDetector(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=self.get_parameter('device').value
        )
        self.get_logger().info('Model loaded successfully')
        
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image, self.get_parameter('input_topic').value, self.image_callback, 10
        )
        
        self.lane_pub = self.create_publisher(Path, self.get_parameter('output_topic').value, 10)
        self.vis_pub = self.create_publisher(Image, self.get_parameter('visualization_topic').value, 10)
        self.weather_pub = self.create_publisher(String, '/perception/weather_condition', 10)
        self.confidence_pub = self.create_publisher(Float32MultiArray, '/perception/lane_confidences', 10)
        
        publish_period = 1.0 / self.get_parameter('publish_rate').value
        self.timer = self.create_timer(publish_period, self.timer_callback)
        
        self.latest_image = None
        self.latest_results = None
        self.frame_count = 0
        self.total_latency = 0.0
        
        self.get_logger().info('Lane detection node initialized')
    
    def image_callback(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {str(e)}')
    
    def timer_callback(self):
        if self.latest_image is None:
            return
        
        t0 = self.get_clock().now()
        results = self.detector.detect(self.latest_image)
        t1 = self.get_clock().now()
        
        latency = (t1 - t0).nanoseconds / 1e6
        self.total_latency += latency
        self.frame_count += 1
        self.latest_results = results
        
        self.publish_lanes(results)
        self.publish_visualization(results)
        self.publish_weather(results)
        self.publish_confidences(results)
        
        if self.frame_count % 100 == 0:
            avg_latency = self.total_latency / self.frame_count
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0
            self.get_logger().info(
                f'Processed {self.frame_count} frames | '
                f'Avg latency: {avg_latency:.1f}ms | FPS: {fps:.1f} | '
                f'Lanes: {len(results["lanes"])} | Weather: {results["weather"]["condition"]}'
            )
    
    def publish_lanes(self, results: dict):
        lanes = results.get('lanes', [])
        for lane in lanes:
            path = Path()
            path.header = Header()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = 'camera_link'
            
            points = lane.get('points', [])
            for pt in points:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = float(pt[0])
                pose.pose.position.y = 0.0
                pose.pose.position.z = float(pt[1])
                path.poses.append(pose)
            
            self.lane_pub.publish(path)
    
    def publish_visualization(self, results: dict):
        if self.latest_image is None:
            return
        vis_image = self.detector.visualize(self.latest_image, results)
        vis_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
        vis_msg.header.stamp = self.get_clock().now().to_msg()
        vis_msg.header.frame_id = 'camera_link'
        self.vis_pub.publish(vis_msg)
    
    def publish_weather(self, results: dict):
        weather = results.get('weather', {})
        msg = String()
        msg.data = f"{weather.get('condition', 'unknown')}|{weather.get('confidence', 0.0):.3f}"
        self.weather_pub.publish(msg)
    
    def publish_confidences(self, results: dict):
        lanes = results.get('lanes', [])
        msg = Float32MultiArray()
        msg.data = [lane.get('confidence', 0.0) for lane in lanes]
        self.confidence_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
