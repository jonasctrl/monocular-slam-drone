#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from PIL.FontFile import WIDTH
from cv_bridge import CvBridge
from depth_estimation_package.msg import RGBWithPose
from depth_estimation_package.msg import Pcd2WithPose
from modules.depth_anything_module import DepthAnythingEstimatorModule
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool
import sensor_msgs.point_cloud2 as pc2
import math

HEIGHT, WIDTH = 144, 256
FOV = 90.0

STRIDE_X = 2
STRIDE_Y = 2
MAX_DEPTH = 100


class DepthEstimationNode:
    def __init__(self):
        rospy.init_node('depth_estimation_node', anonymous=True)

        self.bridge = CvBridge()
        self.depth_estimator_module = DepthAnythingEstimatorModule()

        self.image_sub = rospy.Subscriber('/rgb_with_pose', RGBWithPose, self.image_callback, queue_size=1)

        self.point_cloud_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=1)
        self.pcd_pose_pub = rospy.Publisher('/cam_pcd_pose', Pcd2WithPose, queue_size=1)

        # Compute camera intrinsic parameters
        self.compute_camera_intrinsics()

        rospy.loginfo("Depth estimation node initialized.")

    def compute_camera_intrinsics(self):
        # Convert HFOV to radians
        hfov_rad = FOV * math.pi / 180.0

        # Compute fx and fy (assuming square pixels)
        self.fx = (WIDTH / 2.0) / math.tan(hfov_rad / 2.0)
        self.fy = self.fx  # For square pixels

        # Compute principal point coordinates
        self.cx = WIDTH / 2.0
        self.cy = HEIGHT / 2.0

        rospy.loginfo(f"Camera intrinsics computed: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def rotate_points_x_axis(self, points, angle_degrees=90):
        """
        Rotate points around the X-axis by the specified angle.

        Args:
            points: Nx3 array of points (x, y, z)
            angle_degrees: Rotation angle in degrees
        Returns:
            Rotated points as Nx3 array
        """
        # Convert angle to radians
        angle_rad = np.radians(angle_degrees)

        # Create rotation matrix around X-axis
        rot_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Apply rotation
        return np.dot(points, rot_matrix.T)

    def image_callback(self, msg):
        rospy.loginfo("Received RGBWithPose message")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding='bgr8')
            rospy.loginfo(f"Image size: {cv_image.shape}")

            depth_map = self.depth_estimator_module.generate_depth_map(cv_image)

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

            # NOTE: For debugging purposes
            # Publish the point cloud message
            # self.point_cloud_pub.publish(point_cloud_msg)
            # rospy.loginfo("Published point cloud message")

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def generate_point_cloud(self, depth_map, header):
        header.frame_id = 'camera'

        # Get image dimensions
        height, width = depth_map.shape
        rospy.loginfo(f"Depth map size: {depth_map.shape}")

        # Create meshgrid of pixel coordinates with strides
        u_coords = np.arange(0, width, STRIDE_X)
        v_coords = np.arange(0, height, STRIDE_Y)
        u, v = np.meshgrid(u_coords, v_coords)

        # Flatten the arrays
        u = u.flatten()
        v = v.flatten()
        z = depth_map[v, u].flatten()

        # Filter out invalid depth values
        valid = np.isfinite(z) & (z > 0) & (z < MAX_DEPTH)
        u = u[valid]
        v = v[valid]
        z = z[valid]

        # Compute the 3D coordinates
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        # Stack the coordinates and create points array
        points = np.vstack((x, y, z)).transpose()

        # Rotate points 90 degrees around X-axis
        points_rotated = self.rotate_points_x_axis(points)

        # Create the PointCloud2 message
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        point_cloud_msg = pc2.create_cloud(header, fields, points_rotated)
        return point_cloud_msg

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