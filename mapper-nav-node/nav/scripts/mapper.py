import numpy as np
# import pyquaternion
from scipy.spatial.transform import Rotation as R
# from scipy.ndimage import gaussian_filter, zoom
# import cv2
import csv

# import matplotlib.pyplot as plt

import json
# import pandas as pd
from PIL import Image
import os
# from math import tan, pi

from a_star import a_star_3d

from utils import bresenham3d_check_known_space, depth_img_to_pcd
import inspect

grid_shape = np.array([600, 600, 200]).astype(int)
# grid_st = np.array([0, 0, 0]).astype(int)
# center = np.array([grid_shape[0]//2, grid_shape[1]//2, 50]).astype(int)

def hit():
    print(f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}")

class DataPoint:
    def __init__(self, pcd, pos, qtr=(0, 0, 0, 1), qmod=(0, 0, 0, 1)):
        # (tx, ty, tz) = pos
        # (qx, qy, qz, qw) = qtr
        self.pcd = np.array(pcd)
        self.pos = np.array(pos)
        self.qtr_cof = qtr

        self.q_mod = R.from_quat(qmod)

        self.qtr = R.from_quat(self.qtr_cof)


class VoxArray:
    def __init__(self, resolution=0.08, shape=[600, 600, 200]):
        self.shp = np.array(shape).astype(int)
        self.cntr = np.array([0.5*shape[0],
                              0.5*shape[1],
                              0.25*shape[2]]).astype(int)

        self.res = resolution
        self.bgn = np.array([0, 0, 0]).astype(int)
        self.vox = np.empty(self.shp, dtype=np.int8)
        self.vox.fill(-1)
        self.cam_path = []
        self.cam_path_acc = []
        self.cam_poses = []
        self.nav_path = []

        self.data = []
        self.data_idx = 0
    
        
    def get_known_space_pcd(self):
        x, y, z = np.nonzero(self.vox == 0)
        pcd = np.array([x, y, z]).transpose()
        return pcd
        
    def get_occupied_space_pcd(self):
        x, y, z = np.nonzero(self.vox > 0)
        pcd = np.array([x, y, z]).transpose()
        return pcd

    def navigate(self, start, goal):
        self.nav_path = a_star_3d(self.vox, start, goal)
        print(f"found path={self.nav_path}")

        
    def add_pcd(self, pcd, cam_pt):
        if len(pcd) == 0:
            return 

        cam_pt = np.array(cam_pt).astype(int)

        pcd = np.array(pcd)

        coord = (pcd / self.res).astype(int) + self.cntr
        coord_clp = np.clip(coord, self.bgn, self.shp-1)
        # for point in pcd:
        for point in coord_clp:
            # print(point)
            # coord = (point / self.res).astype(int) + self.cntr
            # coord_clp = np.clip(coord, self.bgn, self.shp-1)
            (x, y, z) = point
            cur = self.vox[x, y, z]
            if cur == -1:
                cur += 1
            if cur < 127:
                cur += 1
            self.vox[x, y, z] = cur
            
            # bresenham3d_check_known_space(cam_pt, point, self.vox)

    def add_pcd_from_datapoint(self, dpt: DataPoint):
        if (len(dpt.pcd) == 0):
            return 
        pcd = dpt.pcd
        # rotate by given quaternion
        pcd = dpt.q_mod.apply(pcd)
        pcd = dpt.qtr.apply(pcd)

        # translate
        pcd = pcd + dpt.pos

        c = (np.array([dpt.pos])/ self.res).astype(int) + self.cntr
        c_acc = np.array([dpt.pos])/ self.res + self.cntr
        c = np.clip(c, self.bgn, self.shp-1).astype(int)
        self.cam_path.append(c[0])

        self.cam_path_acc.append(c_acc[0])
        self.cam_poses.append(dpt.qtr_cof)
        
        self.add_pcd(pcd, c[0])

    def add_pcd_from_depth_image(self, qtr, pos, img, stride, factor, cam_params=None, fov=None):
        point_cloud = depth_img_to_pcd(img, stride, factor, cam_params=cam_params, fov=fov, max_depth=30)
        dpt = DataPoint(point_cloud, pos, qtr)
        self.add_pcd_from_datapoint(dpt)
    
    # def add_pcd_from_file(self, file):
        # pcd = o3d.io.read_point_cloud(file)
        # points = np.asarray(pcd.points)
        # self.add_pcd(points)
        
def find_closest_timestamp(target_timestamp, groundtruth_data):
    # Extract the timestamps from groundtruth data for vectorized operations
    timestamps = np.array([float(row['timestamp']) for row in groundtruth_data])
    
    # Find the index of the closest timestamp
    closest_index = (np.abs(timestamps - target_timestamp)).argmin()
    
    return groundtruth_data[closest_index]
        
def parse_data(n_entries, n_skip):
    # Read depth.csv file
    with open('data/rgbd/depth.csv', 'r') as depth_file:
        depth_reader = csv.DictReader(depth_file)
        depth_data = list(depth_reader)
    
    # Read groundtruth.csv file
    with open('data/rgbd/groundtruth.csv', 'r') as gt_file:
        gt_reader = csv.DictReader(gt_file)
        groundtruth_data = list(gt_reader)

    results = []

    # Camera intrinsic parameters (known from camera)
    fx = 517.3  # Focal length in x
    fy = 516.5  # Focal length in y
    cx = 318.6  # Principal point x
    cy = 255.3  # Principal point y

    # skip = 30
    skip = 15

    # Iterate through depth data
    for i in range(0, min(n_entries, len(depth_data)), n_skip):
        # Read depth image filename and timestamp
        timestamp = float(depth_data[i]['timestamp'])
        filename = depth_data[i]['filename']
        
        print(f"image no. {i} {filename}")
        img = np.array(Image.open(f"data/rgbd/{filename}"))  # Open as grayscale

        factor = 5000  # for the 16-bit PNG files
        # factor = 1  # for the 32-bit float images in ROS bag files

        # Generate the point cloud
        point_cloud = depth_img_to_pcd(img, skip, factor, cam_params=(fx, fy, cx, cy))
        
        # Find the closest ground truth data based on the timestamp
        gt = find_closest_timestamp(timestamp, groundtruth_data)

        # Create a DataPoint object with the point cloud and ground truth data
        results.append(DataPoint(
            point_cloud,
            (float(gt['tx']), float(gt['ty']), float(gt['tz'])),
            (float(gt['qx']), float(gt['qy']), float(gt['qz']), float(gt['qw']))))

    return results

def parse_airsim_data(n_entries, n_skip):
    data_dir = "data"
    dataset_dir = "data"
    # List to store the contents of all JSON files
    metadata_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(os.path.join(data_dir, dataset_dir)):
        print(f"filename={filename}")

        if filename.startswith("metadata_") and filename.endswith(".json"):
            # Construct the full path to the JSON file
            file_path = os.path.join(data_dir, data_dir, filename)
            
            # Open and parse the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
                metadata_list.append(data)

    # Sort the metadata list by the "timestamp" field in ascending order
    metadata_list_sorted = sorted(metadata_list, key=lambda x: x['telemetry']['timestamp'])
    
    results = []
    
    print(f"metadata unsorted len={len(metadata_list)}")
    print(f"metadata len={len(metadata_list_sorted)}")

    fx = 554.26 # Focal length in x
    fy = 415.7  # Focal length in y
    cx = 320    # Principal point x
    cy = 240    # Principal point y
    # factor = 15 # for the 16-bit PNG files
    factor = 1 # for the 16-bit PNG files

    skip = 30
    # skip = 15
    for i in range(0, min(n_entries, len(metadata_list_sorted)), n_skip):
        # Read depth image filename and timestamp
                            
        tl = metadata_list_sorted[i]['telemetry']
        depth_map_path = tl['depth_map_path']
        depth_map_full_path = os.path.join(data_dir, depth_map_path)

        print(f"image no. {i} {depth_map_path}")
        img = np.array(Image.open(depth_map_full_path))  # Open as grayscale

        point_cloud = depth_img_to_pcd(img, skip, factor, cam_params=(fx, fy, cx, cy))
        pos = tl['position']
        ori = tl['orientation']
        
        # results.append(DataPoint(point_cloud, pos['y'], pos['x'], -pos['z'], ori['y'], ori['x'], -ori['z'], ori['w']))
        results.append(DataPoint(point_cloud, (pos['y'], pos['x'], pos['z']),
                                 (ori['x'], ori['y'], ori['z'], ori['w'])))

    return results


def parse_airsim_data_v3(n_entries, n_skip):
    data_dir = "data"
    dataset_dir = "data"
    # List to store the contents of all JSON files
    metadata_list = []

    # Iterate over all files in the directory
    for filename in os.listdir(os.path.join(data_dir, dataset_dir)):

        if filename.startswith("telemetry_") and filename.endswith(".json"):
            print(f"filename={filename}")
            try:
                num_str = filename.split("_")[-1].split(".")[0]
                img_path = os.path.join(data_dir, data_dir, f"depth_map_{num_str}.png")
                img = np.array(Image.open(img_path))  # Open as grayscale
                gt_img_path = os.path.join(data_dir, data_dir, f"ground_truth_depth_map_{num_str}.png")
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
    
    # factor = 0.19583843329253367
    # factor = 1 / 0.19583843329253367
    # factor = 0.1
    # factor = 0.3
    # factor = 0.5
    # factor = 0.7
    factor = 1
    # factor = 1.5
    # factor = 4

    skip = 30
    skip = 5
    # max_depth=15
    max_depth=30

    seek = 0
    for i in range(seek, min(n_entries, len(metadata_list_sorted)), n_skip):
        v = metadata_list_sorted[i]
        img = v["gt_img"]
        # img = v["img"]
        # img = gaussian_filter(img, sigma=4.0)

        
        point_cloud = depth_img_to_pcd(img, skip, factor, fov=v["camera_fov"], max_depth=max_depth)
        # print(f"img.shape={img.shape} pcd.len=pcd.len")

        pos = v['camera_position']
        ori = v['camera_orientation']
        
        results.append(DataPoint(point_cloud, (pos['y'], pos['x'], -pos['z']),
                                 (ori['y'], ori['x'], -ori['z'], ori['w'])
                                 , qmod=[-0.7071, 0, 0, 0.7071]))

    return results




    
