
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from a_star import a_star_3d
from utils import bresenham3d_raycast, clamp, quaternion_from_two_vectors
import nav_config as cfg
# from math import tan, pi
# from d_star import DStar
from drrt import DRRT

from numba import njit

grid_shape = np.array([cfg.map_depth, cfg.map_width, cfg.map_heigth]).astype(int)


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

    
    for point in unique_pcd:
        (x, y, z) = point
        if vox[x, y, z] < cfg.occup_thr:
            changed_pts.append((x, y, z, -1))
        vox[x, y, z] = clamp(vox[x, y, z] + cfg.ray_hit_incr, cfg.occup_min, cfg.occup_max)
        cols = bresenham3d_raycast(cam_pt, point, vox)
        for cur in cols:
            (x, y, z, v) = cur
            if v <= cfg.occup_unkn:
                vox[x, y, z] = cfg.occup_min
                changed_pts.append((x, y, z, 0))
            elif v > cfg.occup_min:
                incr = cfg.ray_miss_incr
                ch_val = clamp(v + incr, cfg.occup_min, cfg.occup_max)
                vox[x, y, z] = ch_val
                if v < cfg.occup_thr and ch_val >= cfg.occup_thr:
                    changed_pts.append((x, y, z, -1))
                elif v >= cfg.occup_thr and ch_val < cfg.occup_thr:
                    changed_pts.append((x, y, z, 0))

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
        self.updated_start_or_goal:bool = False

        # travel_off = np.array(cfg.travel_off)
        # self.goal:tuple = tuple((np.array(self.start) + travel_off).astype(int))
        self.goal:tuple = tuple(self.cntr)

        self.pos:tuple = self.start
        self.pos_acc:tuple = self.start
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
            orig_pos = self.point_from_map(np.array([self.pos_acc]))
            return orig_pos[0].tolist(), self.qtr
        else:
            return self.pos_acc, self.qtr

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

    def set_goal(self, new_goal, update_start=True):
        print(f"set new goal to {new_goal}")
        print(f"cur start={self.start} plan.len={len(self.plan_path)}")
        self.goal = tuple(new_goal)
        self.updated_start_or_goal = True
        if update_start:
            self.start = self.pos 


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

    def point_from_map(self, pt):
        pt_offsetted = pt
        pt_downscaled = pt_offsetted - self.cntr + self.init_off
        pt_upscaled = self.res * pt_downscaled
        # print(f"pt_downscaled={pt_downscaled}")
        # print(f"map={pt} => pt={pt_upscaled}")
        
        return pt_upscaled

    def point_to_map_acc(self, pt):
        pt_downscaled = (pt / self.res)

        # if not self.has_init_off:
            # self.has_init_off = True
            # self.init_off = pt_downscaled.copy()

        pt_offsetted = pt_downscaled + self.cntr - self.init_off 
        pt_clipped = np.clip(pt_offsetted, self.bgn, self.shp-1)
        pt_rounded = np.round(pt_clipped).astype(int)
        
        return pt_rounded

    def point_to_map(self, pt):
        pt_clipped = self.point_to_map_acc(pt)
        pt_rounded = np.round(pt_clipped).astype(int)
        
        return pt_rounded

    def update(self, pcd, cam_pos, cam_qtr, is_glob_fame):
        # Convert camera point to point in local voxel map
        if not self.has_init_off:
            self.has_init_off = True
            self.init_off = np.array([cam_pos]) / self.res
        
        
        (tx, ty, tz) = cam_pos
        cam_pos = (-ty, tx, tz)
        # cam_pos = (ty, tx, tz)
        c = tuple(self.point_to_map(np.array([cam_pos]))[0])
        c_acc = tuple(self.point_to_map_acc(np.array([cam_pos]))[0])

        # NED to ENU
        pcd = [(y, x, -z) for (x, y, z) in pcd]

        # Add position and quaterion to camera positions and orientations
        self.cam_qtrs.append(cam_qtr)
        self.cam_path.append(c)
        self.cam_path_acc.append(c)

        self.pos = c
        self.pos_acc = c_acc
        self.qtr = cam_qtr
        
        
        if not is_glob_fame and len(pcd) > 0:
            qtr = R.from_quat(cam_qtr)
            pcd = np.array(pcd)
            pcd = qtr.apply(pcd)
            pcd = pcd + np.array(cam_pos)
        
        ch_pts = self.add_pcd(pcd, c)

        return ch_pts

        

    def plan(self, ch_pts):
        self.start = self.pos
        if cfg.use_a_star:
            self.plan_a_star(ch_pts)
        elif cfg.use_drrt:
            self.plan_drrt(ch_pts)
        
        
        
    def __is_on_path_soft(self, tolerance:float) -> bool:
        # return True
        if len(self.plan_path) == 0:
            return True
        c = self.pos
        for i, p in enumerate(self.plan_path[:-1]):
            diff = ((c[0] - p[0])**2 + (c[1] - p[1])**2 + (c[2] - p[2])**2) ** 0.5
            if diff <= tolerance:
                print(f"{c} is on the path")
                # self.plan_path = self.plan_path[i:]
                return True

        print(f"{c} is NOT on the path")
        return False

        
    def __is_on_path(self, c):
        if len(self.plan_path) == 0:
            return True
        for i, p in enumerate(self.plan_path[:-1]):
            if p == c:
                print(f"{c} is on the path")
                # self.plan_path = self.plan_path[i:]
                return True

        print(f"{c} is NOT on the path")
        return False


    def __do_obst_interfere(self, path, obst):
        new_obst = [(x, y, z) for x, y, z, v in obst if v == -1]

        common = set(path).intersection(new_obst)
        interf = len(common) > 0
        
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
        
        
    def do_need_new_plan(self, ch_pts):
        if self.updated_start_or_goal:
            return True

        if not self.__is_on_path_soft(tolerance=cfg.path_tolerance):
            return True

        # if not self.__is_on_path(self.pos):
            # return True

        if self.__do_obst_interfere(self.plan_path, ch_pts):
            return True

        return False


    def walk_path(self):
        min_diff = float("inf")
        min_idx = 0
        c = self.pos
        for i, p in enumerate(self.plan_path):
            diff = ((c[0] - p[0])**2 + (c[1] - p[1])**2 + (c[2] - p[2])**2) ** 0.5
            if diff <= min_diff:
                min_diff = diff
                min_idx = i

        print(f"min diff={min_diff} idx={min_idx}")
        if min_idx > 0:
            self.plan_path = self.plan_path[min_idx + 1:]
            self.plan_qtrs = self.plan_qtrs[min_idx + 1:]
        

    def plan_a_star(self, ch_pts):
        if not self.do_need_new_plan(ch_pts):
            self.walk_path()
            # if len(self.plan_path) > 0:
                # self.plan_path = self.plan_path[1:]
            # if len(self.plan_qtrs) > 0:
                # self.plan_qtrs = self.plan_qtrs[1:]
        else:
            print(f"new plan {self.start} => {self.goal}")

            self.plan_path = a_star_3d(self.vox, self.pos, self.goal)
            self.updated_start_or_goal = False
            print(f"found path={self.plan_path}")
            self.__set_plan_qtrs()

            if len(self.plan_path) > 0:
                self.plan_path = self.plan_path[1:]
            if len(self.plan_qtrs) > 0:
                self.plan_qtrs = self.plan_qtrs[1:]
        # print(f"found plan qtrs={self.plan_qtrs}")
        
        

    def plan_drrt(self, ch_pts):
        print(f"pos={self.pos}")
        self.pf.update_obstacles(ch_pts)
        self.pf.update_start(self.pos)
        self.pf.plan()
        self.pf.plan(force_iters=50)
        st_path = self.pf.get_path()
        self.plan_path = st_path
        self.__set_plan_qtrs()
        # print(f"path={self.plan_path}")
        print(f"path.len={len(self.plan_path)}")


        
    def add_pcd(self, pcd, cam_pos):
        return add_pcd_njit(self.vox,
                np.array(pcd),
                np.array(cam_pos, dtype=np.int64),
                self.res,
                self.cntr - self.init_off)




    
