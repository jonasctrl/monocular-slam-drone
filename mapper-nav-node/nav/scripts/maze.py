
from __future__ import annotations
import numpy as np
import random

from typing import Optional, Dict, List, Tuple

from scipy.spatial.transform import Rotation as R

from utils import bresenham3d_raycast, quaternion_from_two_vectors

import nav_config as cfg

def_cam_target_pts = np.array([(cfg.cam_depth, w - cfg.cam_w / 2, h - cfg.cam_h / 2) for w in range(cfg.cam_w) for h in range(cfg.cam_h)])
def_cam_pos = [0,0,0]

class Maze(object):
    def __init__(self, edge=300, height=50, num_obstacles=30):
        self.env_map = self.__generate_voxel_map_with_obstacles((edge, edge, height), num_obstacles=num_obstacles)
        self.cur_pos:tuple = (10,50,5)
        # self.cur_pos = [0,0,0]
        self.cur_qtr:tuple = (0,0,0,1)
        self.pos_history:List[tuple] = [self.cur_pos]
        self.qtr_history:List[tuple] = [self.cur_qtr]
        self.pos_plan:List[tuple] = []
        self.qtr_plan:List[tuple] = []
        self.glob_pcd = []
        self.cam_pcd = []
        self.cam_targets = []

        self.bounds = [(0, edge), (0, edge), (0, height)]


        # TODO move planinng somwhere else
        # goal = (200, 200, 10)
        # goal = (60,50,20)
        # goal = (40,60,4)

        # goal = (100,90,10)
        # start = tuple(self.cur_pos)
        # print(f"searching path from {start} to {goal}")
        # nav_path = a_star_3d(self.env_map, start, goal)

        # nav_qtrs = []
        # for i in range(len(nav_path) - 1):
            # p1 = np.array(nav_path[i])
            # p2 = np.array(nav_path[i+1])
            # p_diff = p2 - p1
            # qtr = quaternion_from_two_vectors(np.array([1, 0, 0]), p_diff)
            # nav_qtrs.append(qtr)
        
        
        # print(f"found path={nav_path}")
        # print(f"nav_qtrs={nav_qtrs}")
        # self.set_plan(nav_path, [self.cur_qtr] + nav_qtrs)


        self.step()
        
    def __generate_voxel_map_with_obstacles(self, shape, num_obstacles, seed=15):
        random.seed(seed)
        (x_size, y_size, z_size) = shape
        voxel_map = np.zeros((x_size, y_size, z_size), dtype=np.int8)
        
        for _ in range(num_obstacles):
            x_start = random.randint(0, x_size - 1)
            y_start = random.randint(0, y_size - 1)

            z1 = random.randint(int(-0.4 * z_size) , int(1.4 * z_size))
            z2 = random.randint(int(-0.4 * z_size) , int(1.4 * z_size))
            z_start = max(0, min(z1, z2, z_size-1))
            z_end = min(z_size - 1, max(z1, z2, 0))
            
            if random.randint(0, 1) % 2 == 0:
                x_size_obs = random.randint(50, 150)  # Width of obstacle (x direction)
                y_size_obs = 1  # Height of obstacle (y direction)
            else:
                x_size_obs = 1  # Width of obstacle (x direction)
                y_size_obs = random.randint(50, 150)  # Height of obstacle (y direction)

            
            x_end = min(x_start + x_size_obs, x_size)
            y_end = min(y_start + y_size_obs, y_size)
            
            # Place the obstacle in the voxel grid (set to 1)
            voxel_map[x_start:x_end, y_start:y_end, z_start:z_end] = 1
        
        voxel_map[:,:,0] = 1
        voxel_map[0,:,:] = 1
        voxel_map[x_size-1,:,:] = 1
        voxel_map[:,0,:] = 1
        voxel_map[:,y_size-1,:] = 1
        return voxel_map

    def get_map_as_pcd(self):
        x, y, z = np.nonzero(self.env_map == 1)
        pcd = np.array([x, y, z]).transpose()
        return pcd

    def get_pose(self):
        return self.cur_pos, self.cur_qtr

    def get_path(self):
        return self.pos_history, self.qtr_history

    def get_plan(self):
        return self.pos_plan, self.qtr_plan

    def get_glob_pcd(self):
        return self.glob_pcd

    def get_cam_pcd(self):
        return self.cam_pcd

    def cam_collisions(self):
        qtr = R.from_quat(self.cur_qtr)
        inv_qtr = qtr.inv()
        pos = np.array(self.cur_pos).astype(int)
        
        cam_targets = def_cam_target_pts.copy()
        cam_targets = cfg.cam_scaling * cam_targets
        cam_targets = qtr.apply(cam_targets)
        cam_targets = pos + cam_targets
        cam_targets = cam_targets.astype(int)

        collided = []
        
        # print(f"cam_targets:{cam_targets}")
        for ctg in cam_targets:
            # col = bresenham3d_raycast(pos, ctg, self.env_map, get_collision=True, update_unkn=False)
            # print(f"casting {pos} to {ctg}")
            cols = bresenham3d_raycast(pos, ctg, self.env_map)
            cols = [(x, y, z) for i, (x, y, z, v) in enumerate(cols) if v != 0 and i + 1 < len(cols) ]
            if len(cols) > 0:
                collided.append(cols[0])


        self.cam_targets = cam_targets
        self.glob_pcd = np.array(collided)
        
        cam_pcd = np.array(collided)
        if len(collided) == 0:
            cam_pcd = []
            return 

        cam_pcd = cam_pcd - pos
        cam_pcd = inv_qtr.apply(cam_pcd)

        self.cam_pcd = cam_pcd.astype(np.float32)
        

        # cam_pcd_kp = qtr.apply(cam_pcd)
        # cam_pcd_kp = cam_pcd_kp + pos
        # self.glob_pcd =  cam_pcd_kp

    def set_plan(self, new_pos_plan:list, new_qtr_plan:list):
        # if len(new_pos_plan) > 0 and self.cur_pos == new_pos_plan[0]:
            # new_pos_plan = new_pos_plan[1:]
            # new_qtr_plan= new_qtr_plan[1:]
        print(f"setting plan from {new_pos_plan[0]}")
        self.pos_plan = new_pos_plan
        self.qtr_plan = new_qtr_plan

    # def navigate(self, start, goal):
        # nav_path = a_star_3d(self.vox, start, goal)
        # print(f"found path={nav_path}")

    def step(self):
        if len(self.pos_plan) == 0:
            # pass
            print(f"No more planned steps")
            # return 

        if len(self.pos_plan):
            # If we are already on the path, do not restart from beginning
            if self.cur_pos in self.pos_plan:
                last_index = len(self.pos_plan) - 1 - self.pos_plan[::-1].index(self.cur_pos)
                print(f"found existing {self.cur_pos} in path")
                for _ in range(last_index + 1):
                    self.pos_plan.pop(0)
                    self.qtr_plan.pop(0)

        if len(self.pos_plan):
            self.cur_pos = self.pos_plan.pop(0)
        if len(self.qtr_plan):
            self.cur_qtr = self.qtr_plan.pop(0)

        print(f"Steping to {self.cur_pos} next={self.pos_plan[0] if len(self.pos_plan) > 0 else 'None'}")

        self.pos_history.append(self.cur_pos) 
        self.qtr_history.append(self.cur_qtr) 

        self.cam_collisions()



