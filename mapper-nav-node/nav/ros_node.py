#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
from std_msgs.msg import Header

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from mapper import mapper_step, mapper_start, VoxArray 

vmap = None

def generate_point_cloud():
    """
    Generate a random point cloud for demonstration purposes.
    
    :return: List of points in the format (x, y, z)
    """
    # Example: Generating 100 random points in 3D space
    num_points = 100
    point_cloud = np.random.rand(num_points, 3)  # 3D points (x, y, z)
    
    # Convert to a list of tuples (x, y, z)
    return [tuple(point) for point in point_cloud]

def make_cam_path_msg(vmap):
    path_msg = Path()

    path_msg.header = Header()
    path_msg.header.frame_id = "map"
    path_msg.header.stamp = rospy.Time.now()

    for pt, qtr in zip(vmap.cam_path_acc, vmap.cam_poses):
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

    return path_msg

def make_occupied_space_msg(vmap):
    points = vmap.get_occupied_space_pcd()
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "map"
    pcd_msg = point_cloud2.create_cloud_xyz32(header, points)
    return pcd_msg

def start_publishing():
    
    vmap = mapper_start()
    
    # Initialize ROS node
    rospy.init_node('mapper_nav', anonymous=True)

    # Create a ROS publisher
    occupied_space_pub = rospy.Publisher('/occupied_space', PointCloud2, queue_size=10)
    cam_path_pub = rospy.Publisher('/cam_path', Path, queue_size=10)

    
    # Set the loop rate (how often the message is published, 10 Hz in this case)
    rate = rospy.Rate(10)
    
    # Frame of reference for the point cloud
    # frame_id = "map"

    while not rospy.is_shutdown():
        mapper_step(vmap)

        pcd_msg = make_occupied_space_msg(vmap)
        cam_path_msg = make_cam_path_msg(vmap)

        occupied_space_pub.publish(pcd_msg)
        cam_path_pub.publish(cam_path_msg)

        # Sleep for the remaining time to maintain the loop rate
        rate.sleep()
        

if __name__ == '__main__':
    try:
        start_publishing()
    except rospy.ROSInterruptException:
        pass

