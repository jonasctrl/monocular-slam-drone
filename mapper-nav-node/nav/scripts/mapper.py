
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from a_star import a_star_3d
from utils import bresenham3d_raycast, depth_img_to_pcd, clamp, quaternion_from_two_vectors
import nav_config as cfg
# from math import tan, pi
from d_star import DStar

from numba import njit

grid_shape = np.array([600, 600, 200]).astype(int)
# grid_st = np.array([0, 0, 0]).astype(int)
# center = np.array([grid_shape[0]//2, grid_shape[1]//2, 50]).astype(int)

# import inspect
# def hit():
    # print(f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}")

@njit
def unique_pcd_njit(pcd):
    new_pcd = []
    for i in range(pcd.shape[0]):
        x1, y1, z1 = pcd[i]
        found = False
        for j in range(i + 1, pcd.shape[0]):
            x2, y2, z2 = pcd[j]
            if x1 == x2 and y1 == y2 and z1 == z2:
                found = True
                break
        if not found:
            new_pcd.append([x1, y1, z1])
    ret_pcd = np.array(new_pcd, dtype=np.int64)
    return ret_pcd


@njit
def add_pcd_njit(vox, pcd, cam_pt, resolution, off):
    # returns list changed voxels and new value as list of (x,y,z,val)
    changed_pts = []
    if len(pcd) == 0:
        return changed_pts

    bgn = np.array([0, 0, 0], dtype=np.int64)
    end = np.array(vox.shape, dtype=np.int64) - 1

    coord = (pcd / resolution + off).astype(np.int64)
    coord_clp = np.clip(coord, bgn, end)
    unique_pcd = unique_pcd_njit(coord_clp)

    # not_ch = 0
    
    for point in unique_pcd:
        (x, y, z) = point
        if vox[x, y, z] < cfg.occup_thr:
            changed_pts.append((x, y, z, -1))
        vox[x, y, z] = cfg.occup_thr
        cols = bresenham3d_raycast(cam_pt, point, vox)
        for cur in cols:
            (x, y, z, v) = cur
            if v == cfg.occup_unkn:
                vox[x, y, z] = cfg.occup_min
                changed_pts.append((x, y, z, 0))
            else:
                incr = cfg.ray_hit_incr if v >= cfg.occup_thr else cfg.ray_miss_incr 
                ch_val = clamp(v + incr, cfg.occup_min, cfg.occup_max)
                vox[x, y, z] = ch_val
                if v < cfg.occup_thr and ch_val >= cfg.occup_thr:
                    changed_pts.append((x, y, z, -1))
                elif v >= cfg.occup_thr and ch_val < cfg.occup_thr:
                    changed_pts.append((x, y, z, 0))
                # else:
                    # not_ch += 1

    # print(f"c={len(changed_pts)} nc={not_ch}")
    return changed_pts



# class DataPoint:
    # def __init__(self, pcd, /spos, qtr=(0, 0, 0, 1), qmod=(0, 0, 0, 1)):
        # # (tx, ty, tz) = pos
        # # (qx, qy, qz, qw) = qtr
        # self.pcd = np.array(pcd)
        # self.pos = np.array(pos)
        # self.qtr_cof = qtr

        # self.q_mod = R.from_quat(qmod)

        # self.qtr = R.from_quat(self.qtr_cof)


class VoxArray:
    def __init__(self, resolution=0.08, shape=[600, 600, 200]):
        self.shp = np.array(shape).astype(int)
        self.cntr = np.array([0.5*shape[0],
                              0.5*shape[1],
                              0.25*shape[2]]).astype(int)

        self.res = resolution
        self.bgn = np.array([0, 0, 0]).astype(int)
        self.vox = np.empty(self.shp, dtype=np.int8)
        self.vox.fill(cfg.occup_unkn)
        self.cam_path = []
        self.cam_path_acc = []
        self.cam_qtrs = []

        self.plan_path = []
        
        self.has_init_off = False
        self.init_off = np.array([0,0,0])
        self.init_pos = self.cntr.copy()
        self.start = self.cntr.copy()

        travel_off = np.array([7, 2, 0])
        self.goal = (self.start + travel_off).astype(int)

        self.pos = self.start.astype(int)
        # print(f"self.pos={self.pos}")
        # print(f"self.goal={self.goal}")
        self.qtr = [0., 0., 0., 1.]

        self.data = []
        self.data_idx = 0

        self.pf = DStar(x_start=int(self.pos[0]), y_start=int(self.pos[1]),
                        x_goal=int(self.goal[0]), y_goal=int(self.goal[1]))
    
    def get_pose(self):
        return self.pos, self.qtr

    def get_start(self):
        return self.start

    def get_goal(self):
        return self.goal

    def get_plan(self):
        return self.plan_path
        
    def get_known_space_pcd(self):
        x, y, z = np.nonzero(self.vox != cfg.occup_unkn)
        pcd = np.array([x, y, z]).transpose()
        return pcd
        
    def get_empty_space_pcd(self):
        x, y, z = np.nonzero((cfg.occup_min <= self.vox) & (self.vox < cfg.occup_thr))
        pcd = np.array([x, y, z]).transpose()
        return pcd
        
    def get_occupied_space_pcd(self):
        x, y, z = np.nonzero(self.vox >= cfg.occup_thr)
        pcd = np.array([x, y, z]).transpose()
        return pcd

    # def navigate(self, start, goal):
        # self.nav_path = a_star_3d(self.vox, start, goal)
        # print(f"found path={self.nav_path}")

    def update(self, pcd, cam_pos, cam_qtr, is_glob_fame):
        # Convert camera point to point in local voxel map
        c = (np.array([cam_pos])/ self.res).astype(int) + self.cntr - self.init_off 
        c = np.clip(c, self.bgn, self.shp-1).astype(int)
        c_acc = np.array([cam_pos])/ self.res + self.cntr - self.init_off
        c_acc = np.clip(c, self.bgn, self.shp-1)
        c_acc = c_acc[0]

        # On first "update" call move camera to center of map
        if not self.has_init_off:
            self.has_init_off = True
            self.init_off = c_acc - self.cntr
            c = c - self.init_off
            c_acc = c_acc - self.init_off


        # Point along the path
        # if len(self.cam_path_acc) > 0:
            # p1 = np.array(self.cam_path_acc[-1])
            # p2 = np.array(cam_pos)
            # p_diff = p2 - p1
            # qtr = quaternion_from_two_vectors(np.array([1, 0, 0]), p_diff)
        # else:
            # qtr = [0, 0, 0, 1]

        # Add position and quaterion to camera positions and orientations
        self.cam_qtrs.append(cam_qtr)
        self.cam_path.append(c[0])
        self.cam_path_acc.append(c_acc)

        self.pos = c[0]
        self.qtr = cam_qtr
        
        
        if not is_glob_fame:
            qtr = R.from_quat(cam_qtr)
            # inv_qtr = qtr.inv()
            pcd = np.array(pcd)
            pcd = qtr.apply(pcd)
            # pcd = inv_qtr.apply(pcd)
            pcd = pcd + np.array(cam_pos)
        
        ch_pts = self.add_pcd(pcd, c[0])
        
        same_level = [p for p in ch_pts if p[2] == self.init_pos[2]]
        print(f"ch_pts={same_level}")
        vals = [(self.vox[x, y, z], v) for (x, y, z, v) in same_level]
        # print(f"ch_pts={ch_pts}")
        print(f"vals={vals}")

        print(f"pos={self.pos}")
        for x, y, z, v in ch_pts:
            if z == self.pos[2]:
                print(f"updating {v}=>{(x,y)}")
                self.pf.update_cell(x, y, v)
        print("replaning")
        self.pf.replan()
        print("getting path")
        st_path = self.pf.get_path()
        self.plan_path = [[s.x, s.y, self.init_pos[2]] for s in st_path]
        print(f"path={self.plan_path}")




        
    def add_pcd(self, pcd, cam_pos):
        return add_pcd_njit(self.vox,
                np.array(pcd),
                np.array(cam_pos, dtype=np.int64),
                self.res,
                self.cntr - self.init_off)

    # def add_pcd_from_datapoint(self, dpt: DataPoint):

        # if (len(dpt.pcd) == 0):
            # return 
        # pcd = dpt.pcd
        # # rotate by given quaternion
        # pcd = dpt.q_mod.apply(pcd)
        # pcd = dpt.qtr.apply(pcd)

        # # translate
        # pcd = pcd + dpt.pos

        # c = (np.array([dpt.pos])/ self.res).astype(int) + self.cntr
        # c_acc = np.array([dpt.pos])/ self.res + self.cntr
        # c = np.clip(c, self.bgn, self.shp-1).astype(int)
        # self.cam_path.append(c[0])

        # self.cam_path_acc.append(c_acc[0])
        # self.cam_qtrs.append(dpt.qtr_cof)
        
        # self.add_pcd(pcd, c[0])

    # def add_pcd_from_depth_image(self, qtr, pos, img, stride, factor, cam_params=None, fov=None):
        # point_cloud = depth_img_to_pcd(img, stride, factor, cam_params=cam_params, fov=fov, max_depth=30)
        # dpt = DataPoint(point_cloud, pos, qtr)
        # self.add_pcd_from_datapoint(dpt)
    



    
