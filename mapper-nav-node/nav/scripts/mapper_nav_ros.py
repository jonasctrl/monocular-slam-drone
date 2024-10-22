#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from nav.msg import Pcd2WithPose, DepthWithPose
import cv2
from cv_bridge import CvBridge
# import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from mapper import VoxArray 
import sys, os

GUI = False

def print_exception(ex):
    exc_type, _ , exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    # print(exc_type, fname, exc_tb.tb_lineno)
    rospy.logerr(f"ERROR: {exc_type} {fname} {exc_tb.tb_lineno}")

def pointcloud2_to_array(msg):
    # Extract the fields from PointCloud2 message
    points = []
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2]])
    return np.array(points)

class MapperNavNode:
    def __init__(self):
        resolution = 2
        self.vmap = VoxArray(resolution=resolution, shape=[600,600,300])
        rospy.init_node('mapper_nav', anonymous=True)

        self.bridge = CvBridge()
        self.occupied_pub = rospy.Publisher('/occupied_space', PointCloud2, queue_size=10)
        self.cam_path_pub = rospy.Publisher('/cam_path', Path, queue_size=10)

        self.depth_sub = rospy.Subscriber('/ground_truth/depth_with_pose', DepthWithPose, self.image_callback)
        self.depth_sub = rospy.Subscriber('/cam_pcd_pose', Pcd2WithPose, self.pcd_pose_callback)

        rospy.loginfo("Mapper-Navigation node initialized.")


    def publish_cam_path_msg(self):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for pt, qtr in zip(self.vmap.cam_path_acc, self.vmap.cam_poses):
            (tx, ty, tz) = pt
            # (qx, qy, qz, qw) = qtr

            # Need to rotate 90 deg in Z axis
            orig_qtr = R.from_quat(qtr)
            z_rot = R.from_euler('z', np.deg2rad(90))
            rot_qtr = z_rot * orig_qtr
            (qx, qy, qz, qw) = rot_qtr.as_quat()
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = rospy.Time.now()

            pose_stamped.pose.position.x = tx
            pose_stamped.pose.position.y = ty
            pose_stamped.pose.position.z = tz

            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw

            path_msg.poses.append(pose_stamped)

        self.cam_path_pub.publish(path_msg)

    def publish_occupied_space_msg(self):
        points = self.vmap.get_occupied_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.occupied_pub.publish(pcd_msg)

    def pcd_pose_callback(self, msg):
        rospy.loginfo("Received Pcd2WithPose message")

        try:
            # depth_map = self.bridge.imgmsg_to_cv2(msg.depth_image, desired_encoding='mono8')
            pcd = pointcloud2_to_array(msg.pcd)
            rospy.loginfo(f"PCD len: {len(pcd)}")

            pos_pt = msg.position
            pos  = [pos_pt.x, pos_pt.y, pos_pt.z]
            # orientation = msg.orientation

            
            # self.vmap.add_pcd_from_depth_image(orientation, position, depth_map, 5, 1, fov=90)

            # if depth_map is a PointCloud in global frame
            self.vmap.add_pcd(pcd, pos)
            
            self.publish_occupied_space_msg()
            self.publish_cam_path_msg()

            # depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')

            # Create and publish DepthWithPose message
            # depth_with_pose_msg = DepthWithPose()
            # depth_with_pose_msg.depth_image = depth_msg
            # depth_with_pose_msg.position = position
            # depth_with_pose_msg.orientation = orientation

            # self.depth_pub.publish(depth_with_pose_msg)
            # rospy.loginfo("Published depth_with_pose message")

        except Exception as e:
            print_exception(e)
            # rospy.logerr(f"Error processing image: {str(e)}")


    def image_callback(self, msg):
        rospy.loginfo("Received DepthithPose message")

        try:
            depth_map = self.bridge.imgmsg_to_cv2(msg.depth_image, desired_encoding='32FC1')
            rospy.loginfo(f"Image size: {depth_map.shape}")

            tx = msg.position.x
            ty = msg.position.y
            tz = msg.position.z
            qx= msg.orientation.x
            qy= msg.orientation.y
            qz= msg.orientation.z
            qw= msg.orientation.w

            factor = 1
            self.vmap.add_pcd_from_depth_image((qx, qy, qz, qw), (tx, ty, tz), depth_map, 5, factor, fov=90)

            self.publish_occupied_space_msg()
            self.publish_cam_path_msg()

        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down depth estimation node...")
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = MapperNavNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
