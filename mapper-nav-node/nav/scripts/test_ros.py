#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
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
import os
import json
from PIL import Image

class TestDataPoint:
    def __init__(self, img, pos, qtr=(0, 0, 0, 1), qmod=(0, 0, 0, 1)):
        # (tx, ty, tz) = pos
        # (qx, qy, qz, qw) = qtr
        self.img = img
        self.pos = pos

        q_mod = R.from_quat(qmod)
        qtr = R.from_quat(qtr)
        res_qtr = q_mod * qtr

        self.qtr = res_qtr.as_quat()

class TestNode:
    def __init__(self):
        rospy.init_node('test_sender', anonymous=True)

        self.bridge = CvBridge()

        self.depth_pub = rospy.Publisher('/ground_truth/depth_with_pose', DepthWithPose, queue_size=10)

        self.data = []
        self.data_idx = 0
        

        rospy.loginfo("Maze map node initialized.")
        self.load_data()
        self.publish_status()

    def parse_airsim_data_v3(self, n_entries, n_skip):
        data_dir = "nav/data"
        dataset_dir = "data"
        # List to store the contents of all JSON files
        metadata_list = []

        print(f"CWD={os.getcwd()}")

        # Iterate over all files in the directory
        for filename in os.listdir(os.path.join(data_dir, dataset_dir)):

            if filename.startswith("telemetry_") and filename.endswith(".json"):
                print(f"filename={filename}")
                try:
                    num_str = filename.split("_")[-1].split(".")[0]
                    img_path = os.path.join(data_dir, dataset_dir, f"depth_map_{num_str}.png")
                    img = np.array(Image.open(img_path))  # Open as grayscale
                    gt_img_path = os.path.join(data_dir, dataset_dir, f"ground_truth_depth_map_{num_str}.png")
                    gt_img = np.array(Image.open(gt_img_path))  # Open as grayscale
                    with open(os.path.join(data_dir, dataset_dir, filename), 'r') as f:
                        data = json.load(f)
                        data["img"] = img
                        data["gt_img"] = gt_img
                        metadata_list.append(data)
                except Exception as e:
                    print(e)
                    continue


        # Sort the metadata list by the "timestamp" field in ascending order
        metadata_list_sorted = sorted(metadata_list, key=lambda x: x['timestamp'])
        
        results = []
        
        # print(f"metadata unsorted len={len(metadata_list)}")
        # print(f"metadata len={len(metadata_list_sorted)}")

        # tx_lst = [v['camera_position']['x'] for v in metadata_list_sorted]
        # ty_lst = [v['camera_position']['y'] for v in metadata_list_sorted]
        # tz_lst = [v['camera_position']['z'] for v in metadata_list_sorted]
        # qx_lst = [v['camera_orientation']['x'] for v in metadata_list_sorted]
        # qy_lst = [v['camera_orientation']['y'] for v in metadata_list_sorted]
        # qz_lst = [v['camera_orientation']['z'] for v in metadata_list_sorted]
        # qw_lst = [v['camera_orientation']['w'] for v in metadata_list_sorted]
        # ticks = range(len(metadata_list))
        # plt.plot(ticks, tx_lst, label="tx")
        # plt.plot(ticks, ty_lst, label="ty")
        # plt.plot(ticks, tz_lst, label="tz")
        # plt.plot(ticks, qx_lst, label="qx")
        # plt.plot(ticks, qy_lst, label="qy")
        # plt.plot(ticks, qz_lst, label="qz")
        # plt.plot(ticks, qw_lst, label="qw")
        # plt.legend()
        # plt.show()

        seek = 0
        for i in range(seek, min(n_entries, len(metadata_list_sorted)), n_skip):
            v = metadata_list_sorted[i]
            img = v["gt_img"]
            # img = v["img"]
            # img = gaussian_filter(img, sigma=4.0)

            
            # print(f"img.shape={img.shape} pcd.len=pcd.len")

            pos = v['camera_position']
            ori = v['camera_orientation']
            
            results.append(TestDataPoint(img, (pos['y'], pos['x'], -pos['z']),
                                     (ori['y'], ori['x'], -ori['z'], ori['w']),
                                         qmod=[-0.7071, 0, 0, 0.7071]))

        return results
    
    
    
    def load_data(self):
        # img_skip = 20))
        # img_skip = 10
        # img_skip = 5
        img_skip = 1

        # img_end = 1
        # img_end = 10
        # img_end = 20
        # img_end = 50
        # img_end = 100
        # img_end = 200
        # img_end = 600
        img_end = 1200
        self.data = self.parse_airsim_data_v3(img_end, img_skip)

    def publish_depth_with_pose(self):
        try:
            if self.data_idx >= len(self.data):
                print("End of Data")
                return 

            print(f"STEP {self.data_idx}")
            dt = self.data[self.data_idx]
            self.data_idx += 1

            # Ensure the depth map is in float32 format (32FC1)
            
            shp = dt.img.shape
            print(f"Shape={shp}")

            img_depth = np.array(dt.img, dtype=np.float32).reshape(shp[0], shp[1])

            # Convert depth image to ROS Image message
            depth_image_msg = self.bridge.cv2_to_imgmsg(img_depth, encoding="32FC1")
            depth_image_msg.header.seq = self.data_idx
            depth_image_msg.header.stamp = rospy.Time.now()
            depth_image_msg.header.frame_id = "camera"

            # Get camera position and orientation
            # position, orientation = self.get_camera_info()
            

            (tx, ty, tz) = dt.pos
            (qx, qy, qz, qw) = dt.qtr

            # Create and publish DepthWithPose message
            depth_with_pose_msg = DepthWithPose()
            depth_with_pose_msg.depth_image = depth_image_msg
            # depth_with_pose_msg.position = position
            # depth_with_pose_msg.orientation = orientation
            depth_with_pose_msg.position.x = tx
            depth_with_pose_msg.position.y = ty
            depth_with_pose_msg.position.z = tz
            depth_with_pose_msg.orientation.x = qx
            depth_with_pose_msg.orientation.y = qy
            depth_with_pose_msg.orientation.z = qz
            depth_with_pose_msg.orientation.w = qw

            self.depth_pub.publish(depth_with_pose_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error: {str(e)}")

        except Exception as e:
            rospy.logerr(f"Error in publish_depth_with_pose: {str(e)}")
        

    def publish_status(self):
        # Independent publisher for status
        rate = rospy.Rate(10)  # 1 Hz
        while not rospy.is_shutdown():
            print(f"Publishing")

            self.publish_depth_with_pose()




            rate.sleep()

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down depth estimation node...")
        # finally:
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        node = TestNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
