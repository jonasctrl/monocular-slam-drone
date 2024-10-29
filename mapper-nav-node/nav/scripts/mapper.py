
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from a_star import a_star_3d
from utils import bresenham3d_raycast, depth_img_to_pcd, clamp, quaternion_from_two_vectors
import nav_config as cfg
# from math import tan, pi
# from d_star import DStar
from drrt import DRRT

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
        self.plan_qtrs = []
        
        self.has_init_off = False
        self.init_off = np.array([0,0,0])
        self.init_pos = self.cntr.copy()
        self.start:tuple = tuple(self.cntr)

        # travel_off = np.array([3, 2, 0])
        # travel_off = np.array([7, 2, 0])
        # travel_off = np.array([20, 5, 2])
        # travel_off = np.array([31, -15, 2])
        # travel_off = np.array([32, 7, 4])
        travel_off = np.array(cfg.travel_off)
        self.goal:tuple = tuple((np.array(self.start) + travel_off).astype(int))

        self.pos:tuple = self.start
        # print(f"self.pos={self.pos}")
        # print(f"self.goal={self.goal}")
        self.qtr = [0., 0., 0., 1.]

        self.data = []
        self.data_idx = 0

        # self.pf = DStar(x_start=int(self.pos[0]), y_start=int(self.pos[1]),
                        # x_goal=int(self.goal[0]), y_goal=int(self.goal[1]))

        # pf_start = (self.pos[0],self.pos[1])
        # pf_goal = (self.goal[0],self.goal[1])
        # print(f"init pf start={pf_start} goal={pf_goal}")
        self.pf = DRRT(start=self.pos, goal=self.goal, shape=(shape[0], shape[1]), step_size=1, max_iter=200, goal_sample_rate=0.2)


    def get_pose(self, orig_coord=True):
        if orig_coord:
            orig_pos = self.point_from_map(np.array([self.pos]))
            return orig_pos[0].tolist(), self.qtr
        else:
            return self.pos, self.qtr

    def get_start(self, orig_coord=True):
        if orig_coord:
            orig_start = self.point_from_map(np.array([self.start]))
            return orig_start[0].tolist()
        else:
            return self.start

    def get_goal(self, orig_coord=True):
        if orig_coord:
            orig_goal = self.point_from_map(np.array([self.goal]))
            return orig_goal[0].tolist()
        else:
            return self.goal

    def get_plan(self, orig_coord=True):
        if len(self.plan_path) == 0:
            # print(f"No plan path")
            return [], []
        if orig_coord:
            orig_path = self.point_from_map(np.array(self.plan_path))
            return orig_path.tolist(), self.plan_qtrs
        else:
            return self.plan_path, self.plan_qtrs
        
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

    def point_from_map(self, pt):
        pt_offsetted = pt
        pt_downscaled = pt_offsetted - self.cntr + self.init_off
        pt_upscaled = self.res * pt_downscaled
        # print(f"pt_downscaled={pt_downscaled}")
        # print(f"map={pt} => pt={pt_upscaled}")
        
        return pt_upscaled

    def point_to_map(self, pt):
        pt_downscaled = (pt / self.res)

        if not self.has_init_off:
            self.has_init_off = True
            self.init_off = pt_downscaled.copy()

        pt_offsetted = pt_downscaled + self.cntr - self.init_off 
        pt_clipped = np.clip(pt_offsetted, self.bgn, self.shp-1)
        pt_rounded = np.round(pt_clipped).astype(int)
        # print(f"pt_ds={pt_downscaled} pt_off={pt_offsetted} init_off={self.init_off}")
        # print(f"pt_clip={pt_clipped} pt_rounded={pt_rounded}")
        # print(f"point={pt} => map={pt_rounded}")
        
        return pt_rounded

    def update(self, pcd, cam_pos, cam_qtr, is_glob_fame):
        # Convert camera point to point in local voxel map
        # c = (np.array([cam_pos])/ self.res).astype(int) + self.cntr - self.init_off 
        # c = np.clip(c, self.bgn, self.shp-1).astype(int)
        
        
        (tx, ty, tz) = cam_pos
        cam_pos = (ty, tx, -tz)

        # (tx, ty, tz) = cam_pos
        # cam_pos = (tx, ty, -tz)
        
        
        # (qx, qy, qz, qw) = cam_qtr
        # cam_qtr = (qy, qx, -qz, qw)

        # (qx, qy, qz, qw) = cam_qtr
        # cam_qtr = (qx, qy, -qz, qw)


        # pcd2= pcd.copy()

        # px = pcd2[:,0]
        # py = pcd2[:,1]
        # pz = pcd2[:,2]

        # pcd[:,0] = py
        # pcd[:,1] = px
        # pcd[:,2] = -pz

        pcd = [(y, x, -z) for (x, y, z) in pcd]
        

        c = tuple(self.point_to_map(np.array([cam_pos]))[0])

        # c = tuple(c)
        # pt_up = self.point_from_map(c) 

        
        # c_acc = np.array([cam_pos])/ self.res + self.cntr - self.init_off
        # c_acc = np.clip(c_acc, self.bgn, self.shp-1)
        # c_acc = c_acc[0]

        # On first "update" call move camera to center of map
        # if not self.has_init_off:
            # self.has_init_off = True
            # self.init_off = c_acc - self.cntr
            # c = c - self.init_off
            # c_acc = c_acc - self.init_off


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
        self.cam_path.append(c)
        self.cam_path_acc.append(c)

        # print(f"settin pos to {tuple(self.point_from_map(c)[0])}")
        self.pos = c
        self.qtr = cam_qtr
        

        
        if not is_glob_fame:
            qtr = R.from_quat(cam_qtr)
            # inv_qtr = qtr.inv()
            pcd = np.array(pcd)
            pcd = qtr.apply(pcd)
            # pcd = inv_qtr.apply(pcd)
            pcd = pcd + np.array(cam_pos)
        
        ch_pts = self.add_pcd(pcd, c)
        
        # self.plan_drrt(cam_pos, ch_pts)
        
        return ch_pts

        

    def plan(self, ch_pts):
        self.start = self.pos
        self.plan_a_star(ch_pts)
        
        
    def __is_on_path(self, c):
        # print(f"is {c} in this path:{self.plan_path}")
        for p in self.plan_path:
            if p == c:
                print(f"{c} is on the path")
                return True

        print(f"{c} is NOT on the path")
        # print(f"path={self.plan_path}")
        return False


    def __do_obst_interfere(self, path, obst):
        new_obst = [(x, y, z) for x, y, z, v in obst if v == -1]
        # del_obst = [(x, y, z) for x, y, z, v in obst if v == 0]

        common = set(path).intersection(new_obst)
        # common = set(path).intersection(obst)

        interf = len(common) > 0

        print(f"interference={interf}")
        
        return interf

        
        
    def __set_plan_qtrs(self):
        nav_qtrs = []
        for i in range(len(self.plan_path) - 1):
            p1 = np.array(self.plan_path[i])
            p2 = np.array(self.plan_path[i+1])
            p_diff = p2 - p1
            qtr = quaternion_from_two_vectors(np.array([1, 0, 0]), p_diff)
            nav_qtrs.append(tuple(qtr))
            if i + i == len(self.plan_path):
                nav_qtrs.append(tuple(qtr))


        self.plan_qtrs = nav_qtrs
        
        
        

    def plan_a_star(self, ch_pts):
        # if self.__is_on_path(self.pos):
            # if not self.__do_obst_interfere(self.plan_path, ch_pts):
                # return 

        # print(f"start={self.start} pos={self.pos}")
        # print(f"new plan {self.start} => {self.goal}")

        print(f"start={tuple(self.point_from_map(self.start)[0])} pos={tuple(self.point_from_map(self.pos)[0])}")
        print(f"new plan {tuple(self.point_from_map(self.start)[0])} => {tuple(self.point_from_map(self.goal)[0])}")

        self.plan_path = a_star_3d(self.vox, self.start, self.goal)
        # print(f"found path={self.plan_path}")
        self.__set_plan_qtrs()
        # print(f"found plan qtrs={self.plan_qtrs}")
        
        

    def plan_drrt(self, cam_pos, ch_pts):

        print(f"received={cam_pos} mapped={self.pos}")
        
        same_level = [p for p in ch_pts if p[2] == self.init_pos[2]]
        print(f"ch_pts={same_level}")
        vals = [(self.vox[x, y, z], v) for (x, y, z, v) in same_level]
        # print(f"ch_pts={ch_pts}")
        print(f"vals={vals}")

        print(f"pos={self.pos}")
        obst_list = [(x,y,v) for (x,y,_,v) in ch_pts]
        self.pf.update_start((self.pos[0], self.pos[1]))
        self.pf.update_obstacles(obst_list)
        # for x, y, z, v in ch_pts:
            # if z == self.pos[2]:
                # print(f"updating {v}=>{(x,y)}")
                # self.pf.update_cell(x, y, v)
        print("replaning")
        self.pf.plan()
        # self.pf.plan(force_iters=50)
        print("getting path")
        st_path = self.pf.get_path()
        self.plan_path = [[s[0], s[1], self.init_pos[2]] for s in st_path[1:]]
        # print(f"path={self.plan_path}")




        
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
    



    
