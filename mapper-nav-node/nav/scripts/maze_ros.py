#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, String
from nav.msg import Pcd2WithPose, DepthWithPose
import cv2
from cv_bridge import CvBridge
# import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

from maze import Maze

GLOB_FRAME = False
# GLOB_FRAME = True

g_num = 0

def pointcloud2_to_array(msg):
    # Extract the fields from PointCloud2 message
    points = []
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2]])
    return np.array(points)

class MazeNode:
    def __init__(self):
        self.maze= Maze(120, 40)
        rospy.init_node('maze', anonymous=True)

        self.bridge = CvBridge()

        self.maze_pub = rospy.Publisher('/maze_map', PointCloud2, queue_size=1)
        self.cam_path_pub = rospy.Publisher('/path_hist', Path, queue_size=1)
        # self.depth_pub = rospy.Publisher('/depth_with_pose', DepthWithPose, queue_size=10)
        self.pcd2_pose_pub = rospy.Publisher('/cam_pcd_pose', Pcd2WithPose, queue_size=10)
        self.cam_pcd_pub = rospy.Publisher('/cam_pcd', PointCloud2, queue_size=1)
        self.pose_pub = rospy.Publisher('/cam_pose', PoseStamped, queue_size=1)
        self.pub_status = rospy.Publisher('/status', String, queue_size=1)
        self.cam_path_pub = rospy.Publisher('/path_hist', Path, queue_size=1)
        self.path_sub = rospy.Subscriber('/plan_path', Path, self.path_callback)

        # Internal state
        self.sensor_data = None
        self.control_command = None

        rospy.loginfo("Maze map node initialized.")
        self.publish_status()


    def publish_cam_path_msg(self):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        cam_poses, cam_qtrs = self.maze.get_path()

        # print(f"cam_qtrs={cam_qtrs}")
        for pt, qtr in zip(cam_poses, cam_qtrs):
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

    def publish_cam_pose_msg(self):
        (pt, qtr) = self.maze.get_pose()
        (tx, ty, tz) = pt
        (qx, qy, qz, qw) = qtr

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


    def publish_maze_map_msg(self):
        points = self.maze.get_map_as_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.maze_pub.publish(pcd_msg)

    def publish_cam_pcd(self):
        points = self.maze.get_glob_pcd()
        # points = self.maze.get_cam_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.cam_pcd_pub.publish(pcd_msg)
        
        
    def publish_pcd2_with_pose(self):
        try:
            # Get camera position and orientation
            (position, orientation) = self.maze.get_pose()
            position = np.array(position, dtype=np.float32)
            (tx, ty, tz) = position
            orientation = np.array(orientation, dtype=np.float32)
            (qx, qy, qz, qw) = orientation
            # print(f"pcd2_with_pose:\t\t{orientation}")


            # print(f"points:{points}")

            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = "map"

            # Create and publish DepthWithPose message
            pcd_with_pose_msg = Pcd2WithPose()
            
            if GLOB_FRAME: 
                points = self.maze.get_glob_pcd()
                pcd_with_pose_msg.is_global_frame.data = True
            else:
                points = self.maze.get_cam_pcd()
                pcd_with_pose_msg.is_global_frame.data = False

            pcd_msg = pc2.create_cloud_xyz32(header, points)

            # global g_num
            # if g_num == 1:
                # print()
                # print()
                # sorted_data = sorted(np.array(points), key=lambda x: (x[0], x[1], x[2]))
                # sorted_data = [d.tolist() for d in sorted_data]
                # s = [[str(e) for e in row] for row in sorted_data]
                # lens = [max(map(len, col)) for col in zip(*s)]
                # fmt = ' '.join('{{:{}}}'.format(x) for x in lens)
                # table = [fmt.format(*row) for row in s]
                # print('\n'.join(table))
                # print()
                # print(f"pos={position}")
                # print(f"qtr={orientation}")
                # print()

            # g_num+=1

            pcd_with_pose_msg.pcd = pcd_msg
            pcd_with_pose_msg.position.x = tx
            pcd_with_pose_msg.position.y = ty
            pcd_with_pose_msg.position.z = tz
            pcd_with_pose_msg.orientation.x = qx
            pcd_with_pose_msg.orientation.y = qy
            pcd_with_pose_msg.orientation.z = qz
            pcd_with_pose_msg.orientation.w = qw

            self.pcd2_pose_pub.publish(pcd_with_pose_msg)

        except Exception as e:
            rospy.logerr(f"Error in publish_pcd2_with_pose[{sys.exc_info()[2].tb_lineno}]: {str(e)}")

    def publish_status(self):
        # Independent publisher for status
        # rate = rospy.Rate(0.5)  # 1 Hz
        rate = rospy.Rate(1)  # 1 Hz
        # rate = rospy.Rate(2)  # 1 Hz
        idx=0
        while not rospy.is_shutdown():
            self.maze.step()
            # print(f"Publishing")

            # status_message = String()
            # status_message.data = "Node is running"
            # self.pub_status.publish(status_message)
            
            if idx % 20 == 0: 
                self.publish_maze_map_msg()

            self.publish_cam_pose_msg()
            self.publish_cam_path_msg()
            self.publish_pcd2_with_pose()
            self.publish_cam_pcd()


            idx += 1

            rate.sleep()

    def path_callback(self, msg):
        # poses = [[p.pose.position.x, p.pose.position.y, p.pose.position.z] \
                # for p in msg.poses]
        # qtrs = [[p.orientation.position.x, p.orientation.position.y, \
                # p.orientation.position.z, p.orientation.w] \
                # for p in msg.poses]

        positions = []
        orientations = []
        
        # Iterate through each pose in the Path message
        for pose_stamped in msg.poses:
            # Extract the position and orientation
            pos = pose_stamped.pose.position
            ori = pose_stamped.pose.orientation
            
            # Append to respective lists
            positions.append((pos.x, pos.y, pos.z))
            orientations.append((ori.x, ori.y, ori.z, ori.w))
            # orientations.append([0., 0., 0., 1.])

        
        # print(f"plan received:{positions}")
        self.maze.set_plan(positions, orientations)

        # self.maze.step()


    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down depth estimation node...")
        # finally:
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = MazeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
