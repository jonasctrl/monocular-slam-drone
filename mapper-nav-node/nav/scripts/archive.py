
import os
import csv
import json
from PIL import Image

# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter, zoom
# import cv2

# import pandas as pd

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

