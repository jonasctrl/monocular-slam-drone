import numpy as np
# import pyquaternion
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter, zoom
import open3d as o3d
import plotly.graph_objects as go
import cv2

import matplotlib.pyplot as plt

import json
import pandas as pd
from PIL import Image
import os
# from math import tan, pi

from a_star import a_star_3d

from utils import bresenham3d_check_known_space, depth_img_to_pcd

grid_st = np.array([0, 0, 0]).astype(int)
grid_shape = np.array([600, 600, 200]).astype(int)
center = np.array([grid_shape[0]//2, grid_shape[1]//2, 50]).astype(int)




class DataPoint:
    def __init__(self, pcd, tx, ty, tz, qx, qy, qz, qw, qmod=[0, 0, 0, 1]):
        self.pcd = np.array(pcd)
        self.pos = np.array([tx, ty, tz])
        
        self.q_mod = R.from_quat(qmod)
        self.qtr = R.from_quat([qx, qy, qz, qw])

class VoxArray:
    def __init__(self, center, resolution, grid_shape, grid_start):
        self.cntr = center
        self.res = resolution
        self.shp = grid_shape
        self.bgn = grid_start
        self.vox = np.empty(self.shp, dtype=np.int8)
        self.vox.fill(-1)
        self.cam_path = []
        self.nav_path = []
    

    def plot(self, show_empty_space=False, use_confidence=False):
        # Obstacles
        x, y, z = np.nonzero(self.vox > 0)

        if use_confidence:
            col_scale = [self.vox[xi, yi, zi] for xi, yi, zi in zip(x, y, z)]
        else:
            col_scale = z
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                           mode='markers',
                                           marker=dict(size=6,
                                                       color=col_scale,
                                                       colorscale='viridis',
                                                       # opacity=0.8
                                                       ),
                                           name='obst')])

        if show_empty_space:
            # Empty space
            x, y, z = np.nonzero(self.vox < 0)
            fig = fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                               marker=dict(size=1, color="blue"), name='known'))


        # Camera path
        px = [p[0] for p in self.cam_path]
        py = [p[1] for p in self.cam_path]
        pz = [p[2] for p in self.cam_path]

        fig.add_trace(go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode='lines',
            line=dict(width=4, color='red'),  # Color and size for the new point
            name='cam_path'
        ))

        fig.add_trace(go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode='markers',
            marker=dict(size=5, color='red'),  # Color and size for the new point
            name='cam_path'
        ))

        # Navigation path
        px = [p[0] for p in self.nav_path]
        py = [p[1] for p in self.nav_path]
        pz = [p[2] for p in self.nav_path]

        fig.add_trace(go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode='lines',
            line=dict(width=6, color='yellow'),  # Color and size for the new point
            name='nav_path'
        ))

        fig.add_trace(go.Scatter3d(
            x=px,
            y=py,
            z=pz,
            mode='markers',
            marker=dict(size=5, color='yellow'),  # Color and size for the new point
            name='cam_path'
        ))

        fig.update_layout(scene=dict(
            # xaxis=dict(range=[grid_st[0], grid_shape[0]], title='x'),
            # yaxis=dict(range=[grid_st[1], grid_shape[1]], title='y'),
            # zaxis=dict(range=[grid_st[2], grid_shape[2]], title='z'),
            aspectmode='data'
        ))

        fig.show()
        
    def navigate(self, start, goal):
        self.nav_path = a_star_3d(self.vox, start, goal)
        print(f"found path={self.nav_path}")

        
    def add_pcd(self, pcd_arr):
        for point in pcd_arr:
            coord = (point / self.res).astype(int) + self.cntr
            coord_clp = np.clip(coord, self.bgn, self.shp-1)
            self.vox[*coord_clp] = 1  # 1 for occupied

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
        c = np.clip(c, self.bgn, self.shp-1)
        self.cam_path.append(c[0])
        

        coord = (pcd / self.res).astype(int) + self.cntr
        coord_clp = np.clip(coord, self.bgn, self.shp-1)
        # for point in pcd:
        for point in coord_clp:
            # print(point)
            # coord = (point / self.res).astype(int) + self.cntr
            # coord_clp = np.clip(coord, self.bgn, self.shp-1)
            cur = self.vox[*point]
            if cur == -1:
                cur += 1
            if cur < 127:
                cur += 1
            self.vox[*point] = cur
            
            bresenham3d_check_known_space(c[0], point, self.vox)

    
    def add_pcd_from_file(self, file):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        self.add_pcd(points)
        
        
        
def find_closest_timestamp(target_timestamp, groundtruth_df):
    # Convert the timestamps to numpy array for vectorized operations
    timestamps = groundtruth_df['timestamp'].values
    # Find the index of the closest timestamp
    closest_index = (np.abs(timestamps - target_timestamp)).argmin()
    return groundtruth_df.iloc[closest_index]

def parse_data(n_entries, n_skip):
    depth_df = pd.read_csv('data/rgbd/depth.csv')
    groundtruth_df = pd.read_csv('data/rgbd/groundtruth.csv')
    results = []
    
    # Camera intrinsic parameters (you need to know these from your camera)
    # fx = 525.0  # Focal length in x
    # fy = 525.0  # Focal length in y
    # cx = 319.5  # Principal point x
    # cy = 239.5  # Principal point y
    
    fx = 517.3  # Focal length in x
    fy = 516.5  # Focal length in y
    cx = 318.6  # Principal point x
    cy = 255.3  # Principal point y

    # skip = 30
    skip = 15


    for i in range(0, min(n_entries, len(depth_df)), n_skip):
        # Read depth image filename and timestamp
        timestamp = depth_df.iloc[i]['timestamp']
        filename = depth_df.iloc[i]['filename']
        
        print(f"image no. {i} {filename}")
        img = np.array(Image.open(f"data/rgbd/{filename}"))  # Open as grayscale

        factor = 5000 # for the 16-bit PNG files
        # factor = 1 # for the 32-bit float images in the ROS bag files

        point_cloud = depth_img_to_pcd(img, skip, factor, cam_params=(fx, fy, cx, cy))
        gt = find_closest_timestamp(timestamp, groundtruth_df)
        
        results.append(DataPoint(point_cloud, gt['tx'], gt['ty'], gt['tz'], gt['qx'], gt['qy'], gt['qz'], gt['qw']))

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
        results.append(DataPoint(point_cloud, pos['y'], pos['x'], pos['z'], ori['x'], ori['y'], ori['z'], ori['w']))

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
        
        results.append(DataPoint(point_cloud, pos['y'], pos['x'], -pos['z'], ori['y'], ori['x'], -ori['z'], ori['w'], qmod=[-0.7071, 0, 0, 0.7071]))
        # results.append(DataPoint(point_cloud, pos['y'], pos['x'], pos['z'], ori['x'], ori['y'], ori['z'], ori['w']))

    return results

# resolution = 5.12
# resolution = 2.56
# resolution = 1.28
resolution = 0.64
# resolution = 0.32
# resolution = 0.16
# resolution = 0.08
# resolution = 0.04
# resolution = 0.02
# resolution = 0.01

# img_skip = 20
# img_skip = 10
img_skip = 5
# img_skip = 1

# img_end = 1
# img_end = 10
# img_end = 20
# img_end = 50
# img_end = 100
# img_end = 200
# img_end = 600
img_end = 1200
start = (322, 309, 104)
goal = (283, 307, 102)

# start = (320, 281, 104)
# goal = (312, 289, 102)

vmap = VoxArray(center, resolution, grid_shape, grid_st)
data_arr = parse_airsim_data_v3(img_end, img_skip)
# data_arr = parse_airsim_data(img_end, img_skip)
# data_arr = parse_data(img_end, img_skip)
for i in range(len(data_arr)):
    vmap.add_pcd_from_datapoint(data_arr[i])        
    
# vmap.navigate(start, goal)
# vmap.plot()
vmap.plot(use_confidence=True)


