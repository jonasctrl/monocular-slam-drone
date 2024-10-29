#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from depth_estimation_package.msg import RGBWithPose
from depth_estimation_package.msg import Pcd2WithPose
from modules.depth_anything_module import DepthAnythingEstimatorModule
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool
import sensor_msgs.point_cloud2 as pc2
import tf
import math

GUI = True


class DepthEstimationNode:
    def __init__(self):
        rospy.init_node('depth_estimation_node', anonymous=True)

        self.bridge = CvBridge()
        self.depth_estimator_module = DepthAnythingEstimatorModule()

        self.image_sub = rospy.Subscriber('/rgb_with_pose', RGBWithPose, self.image_callback)
        # Update publisher to use new message type and topic
        self.point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)
        self.pcd_pose_pub = rospy.Publisher('/cam_pcd_pose', Pcd2WithPose, queue_size=1)

        # Initialize the TF broadcaster
        self.tf_broadcaster = tf.TransformBroadcaster()

        # Camera settings
        self.image_width = rospy.get_param('~image_width', 256)
        self.image_height = rospy.get_param('~image_height', 144)
        self.hfov_degrees = rospy.get_param('~hfov_degrees', 90.0)

        # Compute camera intrinsic parameters
        self.compute_camera_intrinsics()

        rospy.loginfo("Depth estimation node initialized.")

    def compute_camera_intrinsics(self):
        # Convert HFOV to radians
        hfov_rad = self.hfov_degrees * math.pi / 180.0

        # Compute fx and fy (assuming square pixels)
        self.fx = (self.image_width / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = self.fx  # For square pixels

        # Compute principal point coordinates
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

        rospy.loginfo(f"Camera intrinsics computed: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def image_callback(self, msg):
        rospy.loginfo("Received RGBWithPose message")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding='bgr8')
            rospy.loginfo(f"Image size: {cv_image.shape}")

            depth_map = self.depth_estimator_module.generate_depth_map(cv_image)

            if GUI:
                depth_display = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imshow("Depth Map", depth_display)
                cv2.waitKey(1)

            # Generate point cloud from depth map
            point_cloud_msg = self.generate_point_cloud(depth_map, msg.rgb_image.header)

            # Create and publish Pcd2WithPose message
            pcd_pose_msg = Pcd2WithPose()
            pcd_pose_msg.pcd = point_cloud_msg

            # Copy position and orientation from input message
            pcd_pose_msg.position = msg.position
            pcd_pose_msg.orientation = msg.orientation

            # Set is_global_frame to False
            pcd_pose_msg.is_global_frame = Bool(False)

            # Publish the combined message
            self.pcd_pose_pub.publish(pcd_pose_msg)
            rospy.loginfo("Published Pcd2WithPose message")

            # Publish the point cloud message
            self.point_cloud_pub.publish(point_cloud_msg)
            rospy.loginfo("Published point cloud message")

            # Broadcast the transform between 'map' and 'camera' frames
            self.broadcast_transform(msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def generate_point_cloud(self, depth_map, header):
        # Set the frame_id to 'camera' or your desired frame
        header.frame_id = 'camera'  # Ensure this matches the frame in your TF tree

        # Get image dimensions
        height, width = depth_map.shape
        rospy.loginfo(f"Depth map size: {depth_map.shape}")

        # Create meshgrid of pixel coordinates
        u_coords = np.arange(width)
        v_coords = np.arange(height)
        u, v = np.meshgrid(u_coords, v_coords)

        # Flatten the arrays
        u = u.flatten()
        v = v.flatten()
        z = depth_map.flatten()

        # Filter out invalid depth values
        valid = np.isfinite(z) & (z > 0)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        # Compute the 3D coordinates
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # Stack the coordinates
        points = np.vstack((x, y, z)).transpose()

        # Create the PointCloud2 message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        point_cloud_msg = pc2.create_cloud(header, fields, points)
        return point_cloud_msg

    def broadcast_transform(self, msg):
        # Extract position and orientation from the RGBWithPose message
        position = (msg.position.x, msg.position.y, msg.position.z)
        orientation = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

        # Broadcast the transform from 'map' to 'camera'
        self.tf_broadcaster.sendTransform(
            position,
            orientation,
            rospy.Time.now(),
            'camera',  # Child frame
            'map'  # Parent frame
        )

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down depth estimation node...")
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = DepthEstimationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass