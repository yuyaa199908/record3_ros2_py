import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException, ExternalShutdownException
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

import logging
# msgs
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import Point,Pose, Vector3, PoseStamped
# from shape_msgs.msg import Mesh, MeshTriangle
# from visualization_msgs.msg import Marker
# from std_msgs.msg import ColorRGBA

# add
import numpy as np
from cv_bridge import CvBridge
import cv2
from builtin_interfaces.msg import Time
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation
from collections import deque
from record3d import Record3DStream
import threading
import open3d as o3d
import pypcd4
import struct


class Record3DCameraNode(Node):
    def __init__(self):
        super().__init__('record3d_camera_node')
        self.init_param()
        self.init_value()
        self.connect_to_device()

        self.pub_depth =  self.create_publisher(Image, '/output_depth', 10)
        self.pub_color =  self.create_publisher(Image, '/output_color', 10)
        self.pub_color_info =  self.create_publisher(CameraInfo, '/output_color_info', 10)
        self.pub_confidence =  self.create_publisher(Image, '/output_confidence', 10)
        self.pub_pose = self.create_publisher(PoseStamped, '/output_pose', 10)
        self.pub_cloud =  self.create_publisher(PointCloud2, '/output_cloud', 10)
        

        self.start_processing_stream()

    def init_param(self):
        self.declare_parameter('depth.frame_id', "camera")
        self.declare_parameter('color.frame_id', "camera")
        self.declare_parameter('confidence.pub_flag', True)
        self.declare_parameter('confidence.frame_id', "camera")
        self.declare_parameter('pose.pub_flag', True)
        self.declare_parameter('pose.frame_id', "camera")
        self.declare_parameter('cloud.pub_flag', True)
        self.declare_parameter('cloud.frame_id', "camera")

        self.depth_frame_id = self.get_parameter('depth.frame_id').get_parameter_value().string_value
        self.color_frame_id = self.get_parameter('color.frame_id').get_parameter_value().string_value
        self.confidence_pub_flag = self.get_parameter('confidence.pub_flag').get_parameter_value().bool_value
        self.confidence_frame_id = self.get_parameter('confidence.frame_id').get_parameter_value().string_value
        self.pose_pub_flag = self.get_parameter('pose.pub_flag').get_parameter_value().bool_value
        self.pose_frame_id = self.get_parameter('pose.frame_id').get_parameter_value().string_value
        self.cloud_pub_flag = self.get_parameter('cloud.pub_flag').get_parameter_value().bool_value
        self.cloud_frame_id = self.get_parameter('cloud.frame_id').get_parameter_value().string_value

    def init_value(self):
        self.event = threading.Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.dev_idx = 0
        self.frame_id = "map"
        self.scale = 1.0
        
    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')
        rclpy.shutdown()
        
    def connect_to_device(self):
        self.get_logger().info(f"Searching for devices")
        devs = Record3DStream.get_connected_devices()
        self.get_logger().info(f"{len(devs)} device(s) found")
        for dev in devs:
            self.get_logger().info(f'ID: {dev.product_id}\tUDID: {dev.udid}')

        if len(devs) <= self.dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(self.dev_idx))
        dev = devs[self.dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])

    def start_processing_stream(self):
        while rclpy.ok():
            self.event.wait()  # Wait for new frame to arrive
            
            coeffs = self.session.get_intrinsic_mat()
            # intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())

            # pub depth
            depth = self.session.get_depth_frame()
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
            msg_depth = CvBridge().cv2_to_imgmsg(depth, encoding="passthrough")
            msg_depth.header.stamp = self.get_clock().now().to_msg()
            msg_depth.header.frame_id = self.depth_frame_id
            self.pub_depth.publish(msg_depth)

            #pub color
            rgb = self.session.get_rgb_frame()
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                rgb = cv2.flip(rgb, 1)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            msg_color = CvBridge().cv2_to_imgmsg(rgb, encoding="bgr8")
            msg_color.header.stamp = self.get_clock().now().to_msg()
            msg_color.header.frame_id = self.color_frame_id
            self.pub_color.publish(msg_color)

            #pub color info
            h,w = rgb.shape[:2] 
            msg_color_info = self.create_camerainfo(coeffs,h,w)
            self.pub_color_info.publish(msg_color_info)

            #pub confidence
            if self.confidence_pub_flag:
                confidence = self.session.get_confidence_frame()
                if confidence.shape[0] > 0 and confidence.shape[1] > 0:
                    msg_confidence = CvBridge().cv2_to_imgmsg(confidence * 100, encoding="passthrough")
                    msg_confidence.header.stamp = self.get_clock().now().to_msg()
                    msg_confidence.header.frame_id = self.confidence_frame_id
                    self.pub_confidence.publish(msg_confidence)
            # pub pose
            if self.pose_pub_flag:
                camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])
                msg_pose = self.create_posestamped(camera_pose)
                self.pub_pose.publish(msg_pose)

            # pub cloud
            if self.cloud_pub_flag:
                self.process(rgb, depth, coeffs)
            
            self.event.clear()

    def create_camerainfo(self,coeffs,h,w):
        msg = CameraInfo()
        msg.header.frame_id = self.color_frame_id
        # now = self.get_clock().now()
        # msg.header.stamp = Time(sec=now.seconds_nanoseconds()[0], nanosec=now.seconds_nanoseconds()[1])
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = h
        msg.width = w
        msg.distortion_model = "plumb_bob"  #TODO: Check iphone camera
        msg.d = [0., 0., 0., 0., 0.]
        msg.k = [coeffs.fx, 0., coeffs.tx, 0., coeffs.fy, coeffs.ty, 0., 0., 1.]
        msg.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        msg.p = [coeffs.fx, 0., coeffs.tx, 0., 0., coeffs.fy, coeffs.ty, 0., 0., 0., 1., 0.]
        # msg.binning_x = 0
        # msg.binning_y = 0
        # msg.roi = 
        return msg

    def create_posestamped(self,data):
        msg = PoseStamped()
        msg.header.frame_id = self.pose_frame_id
        # now = self.get_clock().now()
        # msg.header.stamp = Time(sec=now.seconds_nanoseconds()[0], nanosec=now.seconds_nanoseconds()[1])
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.orientation.x = data.qx
        msg.pose.orientation.y = data.qy
        msg.pose.orientation.z = data.qz
        msg.pose.orientation.w = data.qw
        msg.pose.position.x = data.tx
        msg.pose.position.y = data.ty
        msg.pose.position.z = data.tz
        return msg

    def process(self,input_rgb, input_d, coeffs):
        # resize
        input_d = cv2.resize(input_d * 1000, dsize=None, fx=self.scale, fy=self.scale,interpolation=cv2.INTER_NEAREST)
        height, width = input_d.shape
        scale_rgb2d = input_d.shape[0]/input_rgb.shape[0]
        input_rgb = cv2.resize(input_rgb,(width, height),interpolation=cv2.INTER_NEAREST)
        
        fx_d = coeffs.fx * scale_rgb2d
        fy_d = coeffs.fy * scale_rgb2d
        cx_d = coeffs.tx * scale_rgb2d
        cy_d = coeffs.tx * scale_rgb2d

        # create cloud
        color = o3d.geometry.Image(input_rgb)
        depth = o3d.geometry.Image(input_d)#.astype(np.uint16))

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx_d,fy_d, cx_d, cy_d
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False
        )
        pcd_o3d = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, pinhole_camera_intrinsic,
            project_valid_depth_only=True
        )

        # # Convert to Open3D.PointCLoud:
        pcd_o3d.transform( [[0,0, 1,0], 
                            [-1, 0, 0, 0], 
                            [0, -1, 0, 0], 
                            [0, 0, 0, 1]])

        msg_out = self.convert_o3d_to_ros2(pcd_o3d)

        self.pub_cloud.publish(msg_out)

    def convert_o3d_to_ros2(self, pcd_o3d):
        # TODO: fix convert_rgb_array_to_float
        try:
            header = Header()
            # now = self.get_clock().now()
            # header.stamp = Time(sec=now.seconds_nanoseconds()[0], nanosec=now.seconds_nanoseconds()[1])
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.cloud_frame_id
            rgb_float_array = self.convert_rgb_array_to_float(pcd_o3d.colors)
            arr = np.concatenate([np.asarray(pcd_o3d.points) ,rgb_float_array.reshape((-1,1))],1)
            pc = pypcd4.PointCloud.from_xyzrgb_points(arr) #PointCloud.from_points(arr, fields, types)
            out_msg = pc.to_msg(header)
        except:
            out_msg = self.create_empty_cloud_msg()

        return out_msg

    def create_empty_cloud_msg(self):
        header = Header()
        # now = self.get_clock().now()
        # header.stamp = Time(sec=now.seconds_nanoseconds()[0], nanosec=now.seconds_nanoseconds()[1])
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.cloud_frame_id
        msg_out = PointCloud2()
        msg_out.header = header
        msg_out.height = 1
        msg_out.width = 0
        msg_out.fields =[
                                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                                    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
                                ]
        msg_out.is_bigendian = False
        msg_out.point_step = 16
        msg_out.row_step = 0
        msg_out.is_dense = True
        msg_out.data = bytes([])
        return msg_out   

    def convert_rgb_array_to_float(self, rgb_array):
        """
        Convert an array of RGB values (0.0 to 1.0) to an array of single float32 RGB values.
        """
        rgb_float_array = np.apply_along_axis(lambda x: self.rgb_to_float(x[2], x[1], x[0]), 1, rgb_array)
        return rgb_float_array

    def rgb_to_float(self, r, g, b):
        """
        Convert separate R, G, B values (0.0 to 1.0) to a single float32 RGB value.
        """
        # Ensure the RGB values are in the range 0 to 255
        r = int(r * 255.0)
        g = int(g * 255.0)
        b = int(b * 255.0)
        
        # Combine the RGB values into a single 32-bit integer
        rgb_int = (r << 16) | (g << 8) | b
        
        # Pack this integer into a float32
        rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
        return rgb_float


def main():
    rclpy.init()
    node = Record3DCameraNode()
    try:
    # rclpy.spin()は無限ループになるが、shutdownが呼び出されると終了する
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # ノードを安全に破棄
        rclpy.shutdown()     # ROS 2をシャットダウン

if __name__ == '__main__':
    main()