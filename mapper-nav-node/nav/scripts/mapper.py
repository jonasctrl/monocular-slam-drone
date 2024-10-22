
import numpy as np
from scipy.spatial.transform import Rotation as R

from a_star import a_star_3d
from utils import bresenham3d_raycast, depth_img_to_pcd, clamp
import nav_config as cfg
# from math import tan, pi


grid_shape = np.array([600, 600, 200]).astype(int)
# grid_st = np.array([0, 0, 0]).astype(int)
# center = np.array([grid_shape[0]//2, grid_shape[1]//2, 50]).astype(int)

# import inspect
# def hit():
    # print(f"{inspect.currentframe().f_code.co_name}:{inspect.currentframe().f_lineno}")

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
        x, y, z = np.nonzero(self.vox != cfg.occup_unkn)
        pcd = np.array([x, y, z]).transpose()
        return pcd
        
    def get_occupied_space_pcd(self):
        x, y, z = np.nonzero(self.vox >= cfg.occup_thr)
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
            
            self.vox[x, y, z] = cfg.occup_thr
            cols = bresenham3d_raycast(cam_pt, point, self.vox)
            for (x, y, z, v) in cols:
                if v == cfg.occup_unkn:
                    self.vox[x, y, z] = cfg.occup_min
                else:
                    incr = cfg.ray_hit_incr if v >= cfg.occup_thr else cfg.ray_miss_incr 
                    self.vox[x, y, z] = clamp(v + incr, cfg.occup_min, cfg.occup_max)
            
            # for dx in [-1, 0, 1]:
                # for dy in [-1, 0, 1]:
                    # for dz in [-1, 0, 1]:
                        # new_x, new_y, new_z = x + dx, y + dy, z + dz
                        # if in_bounds(self.vox, new_x, new_y, new_z):
                            # self.vox[new_x, new_y, new_z] = cur

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
        



    
