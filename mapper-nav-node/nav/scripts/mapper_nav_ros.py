#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Header
from nav.msg import Pcd2WithPose, DepthWithPose
import cv2
from cv_bridge import CvBridge
# import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from mapper import VoxArray 
import sys, os

import nav_config as cfg


g_num = 0

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
        self.vmap = VoxArray(resolution=cfg.map_resolution, shape=[600,600,300])
        rospy.init_node('mapper_nav', anonymous=True)

        self.bridge = CvBridge()
        self.occupied_pub = rospy.Publisher('/occupied_space', PointCloud2, queue_size=3)
        self.empty_pub = rospy.Publisher('/empty_space', PointCloud2, queue_size=3)
        self.cam_path_pub = rospy.Publisher('/cam_path', Path, queue_size=10)
        self.plan_path_pub = rospy.Publisher('/plan_path', Path, queue_size=10)
        self.pose_pub = rospy.Publisher('/map_pose', PoseStamped, queue_size=10)

        self.start_pub = rospy.Publisher('/nav_start', PointStamped, queue_size=10)
        self.goal_pub = rospy.Publisher('/nav_goal', PointStamped, queue_size=10)

        self.depth_sub = rospy.Subscriber('/ground_truth/depth_with_pose', DepthWithPose, self.image_callback)
        self.depth_sub = rospy.Subscriber('/cam_pcd_pose', Pcd2WithPose, self.pcd_pose_callback)

        rospy.loginfo("Mapper-Navigation node initialized.")


    def publish_cam_path_msg(self):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        # print(f"cam_poses={self.vmap.cam_poses}")
        for pt, qtr in zip(self.vmap.cam_path_acc, self.vmap.cam_qtrs):
            (tx, ty, tz) = pt
            (qx, qy, qz, qw) = qtr

            # Need to rotate 90 deg in Z axis
            # orig_qtr = R.from_quat(qtr)
            # z_rot = R.from_euler('z', np.deg2rad(90))
            # rot_qtr = z_rot * orig_qtr
            # (qx, qy, qz, qw) = rot_qtr.as_quat()
            
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

    def publish_plan_path_msg(self):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        # print(f"cam_poses={self.vmap.cam_poses}")
        path = self.vmap.get_plan()
        for pt in path:
            (tx, ty, tz) = pt

            # Need to rotate 90 deg in Z axis
            # orig_qtr = R.from_quat(qtr)
            # z_rot = R.from_euler('z', np.deg2rad(90))
            # rot_qtr = z_rot * orig_qtr
            # (qx, qy, qz, qw) = rot_qtr.as_quat()
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = rospy.Time.now()

            pose_stamped.pose.position.x = tx
            pose_stamped.pose.position.y = ty
            pose_stamped.pose.position.z = tz

            pose_stamped.pose.orientation.x = 0
            pose_stamped.pose.orientation.y = 0
            pose_stamped.pose.orientation.z = 0
            pose_stamped.pose.orientation.w = 1

            path_msg.poses.append(pose_stamped)

        self.plan_path_pub.publish(path_msg)

    def publish_map_pose_msg(self):
        (pt, qtr) = self.vmap.get_pose()
        (tx, ty, tz) = pt
        (qx, qy, qz, qw) = qtr

        # print(f"cur\t qtr={qtr}")
        # print(f"lpt\t qtr={self.vmap.cam_qtrs[-1]}")

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

        self.pose_pub.publish(pose_stamped)


    def publish_start_msg(self):
        (x, y, z) = self.vmap.get_start()
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"  # Adjust to your frame of reference
        point_msg.point = Point(x, y, z)
        self.start_pub.publish(point_msg)

    def publish_goal_msg(self):
        (x, y, z) = self.vmap.get_goal()
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"  # Adjust to your frame of reference
        point_msg.point = Point(x, y, z)
        self.goal_pub.publish(point_msg)

    def publish_occupied_space_msg(self):
        points = self.vmap.get_occupied_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.occupied_pub.publish(pcd_msg)

    def publish_empty_space_msg(self):
        points = self.vmap.get_empty_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.empty_pub.publish(pcd_msg)

    def pcd_pose_callback(self, msg):
        pcd = pointcloud2_to_array(msg.pcd)

        pos_pt = msg.position
        pos  = [pos_pt.x, pos_pt.y, pos_pt.z]

        qtr_pt = msg.orientation
        qtr  = [qtr_pt.x, qtr_pt.y, qtr_pt.z, qtr_pt.w]

        is_glob_fame = msg.is_global_frame.data

        # global g_num
        # if g_num == 0:
            # print()
            # print()
            # # sorted_arr = sorted(pcd)
            # sorted_data = sorted(pcd, key=lambda x: (x[0], x[1], x[2]))
            # sorted_data = [d.tolist() for d in sorted_data]
            # s = [[str(e) for e in row] for row in sorted_data]
            # lens = [max(map(len, col)) for col in zip(*s)]
            # fmt = ' '.join('{{:{}}}'.format(x) for x in lens)
            # table = [fmt.format(*row) for row in s]
            # print('\n'.join(table))
            # print()
            # print(f"pos={pos}")
            # print(f"qtr={qtr}")
            # print()
        # g_num+=1

        self.vmap.update(pcd, pos, qtr, is_glob_fame)
        
        self.publish_start_msg()
        self.publish_goal_msg()
        self.publish_plan_path_msg()
        self.publish_map_pose_msg()
        self.publish_occupied_space_msg()
        self.publish_empty_space_msg()
        self.publish_cam_path_msg()



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