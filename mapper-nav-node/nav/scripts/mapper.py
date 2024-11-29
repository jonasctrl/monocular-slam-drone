
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import random

from a_star import a_star_3d
from utils import bresenham3d_raycast, clamp, quaternion_from_two_vectors
import nav_config as cfg
from drrt import DRRT

from numba import njit

grid_shape = np.array([cfg.map_depth, cfg.map_width, cfg.map_heigth]).astype(int)

@njit
def get_empty_space_pcd(vox):
    x, y, z = np.nonzero((cfg.occup_min <= vox) & (vox < cfg.occup_thr))
    pcd = np.vstack((x, y, z)).transpose()
    return pcd

@njit
def get_occupied_space_pcd(vox):
    x, y, z = np.nonzero(vox >= cfg.occup_thr)
    pcd = np.vstack((x, y, z)).transpose()
    vals = []
    for (x, y, z) in pcd:
        vals.append(vox[x, y, z])
    values = np.array(vals, dtype=np.int8)
    return pcd, values

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

def precompile():
    print(f"precompile ... ", end="")
    vox = np.empty((6,6,6), dtype=np.int8)
    vox.fill(-128)
    vals = np.random.randint(6, 12, size=10, dtype=np.int8)
    for v in vals:
        x = random.randint(0, vox.shape[0]-1)
        y = random.randint(0, vox.shape[1]-1)
        z = random.randint(0, vox.shape[2]-1)
        vox[x, y, z] = v

    get_occupied_space_pcd(vox)
    get_empty_space_pcd(vox)

    pcd = [(1,2,3), (0,2,2), (1,2,1), (0,0,2)]
    add_pcd_njit(vox,
        np.array(pcd),
        np.array((1,2,0), dtype=np.int64),
        2.0,
        np.array((2,1,0)))

    a_star_3d(vox, (1,1,2), (3,4,4), (1,3))

    print(f"done")

def make_fat_path(path):
    fat_path = set()
    for p in path:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    fat_path.add((p[0] + i, p[1] + j, p[2] + k))
    return fat_path


class VoxArray:
    def __init__(self, resolution:float, shape:tuple):
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

        self.fat_path = set()
        self.plan_path = []
        self.plan_qtrs = []
        self.plan_unf = False
        self.replan_cnt = 0
        
        self.has_init_off = False
        self.init_off = np.array([0.,0.,0.])
        self.init_off_round = np.array([0,0,0])
        # self.init_pos = self.cntr.copy()
        self.start:tuple = tuple(self.cntr)
        self.updated_start_or_goal:bool = False

        # goal_off = np.array(cfg.goal_off)
        # self.goal:tuple = tuple((np.array(self.start) + goal_off).astype(int))
        if len(cfg.goal_off) == 3:
            self.goal:tuple = ( \
                    int(round(self.cntr[0] + cfg.goal_off[0] / cfg.map_resolution)), \
                    int(round(self.cntr[1] + cfg.goal_off[1] / cfg.map_resolution)), \
                    int(round(self.cntr[2] + cfg.goal_off[2] / cfg.map_resolution)))
            self.updated_start_or_goal = True
        elif len(cfg.goal_off_vox) == 3:
            self.goal:tuple = ( \
                    self.cntr[0] + cfg.goal_off_vox[0], \
                    self.cntr[1] + cfg.goal_off_vox[1], \
                    self.cntr[2] + cfg.goal_off_vox[2])
            self.updated_start_or_goal = True
        else:
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
        if cfg.use_drrt:
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
        return get_empty_space_pcd(self.vox)
        
    def get_occupied_space_pcd(self):
        return get_occupied_space_pcd(self.vox)

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
            self.init_off_round = np.round(self.init_off[0]).astype(int)
        
        c = tuple(self.point_to_map(np.array([cam_pos]))[0])
        c_acc = tuple(self.point_to_map_acc(np.array([cam_pos]))[0])

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
        if cfg.use_drrt:
            self.plan_drrt(ch_pts)
        elif cfg.use_a_star:
            self.plan_a_star(ch_pts)
        
    def __is_not_on_path_soft(self, tolerance:float) -> bool:
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


    def __is_obst_interfere(self, path, obst):
        new_obst = [(x, y, z) for x, y, z, v in obst if v == -1]

        common = set(path).intersection(set(new_obst))
        interf = len(common) > 0

        return interf


    def __is_unfinished_plan(self):
        return self.plan_unf and len(self.plan_path) < cfg.unf_plan_limit

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
        # Was start or goal location uptated manally?
        if self.updated_start_or_goal:
            return True

        # Do we need to replan because previous path planning was cut short
        # due to iteration limit, and that (unfinished) path is nearing to
        # the end. Thus, we need to replan again in order to not to stop prematurely 
        if self.__is_unfinished_plan():
            return True

        # Is drone not on the planned path (with given voxel tolerence)?
        if not self.__is_not_on_path_soft(tolerance=cfg.path_drift_tolerance):
            return True

        # Are newly found obstaces on the 'fat' path (path whith one voxel clearance)?
        if self.__is_obst_interfere(self.plan_path, ch_pts):
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

        # print(f"min diff={min_diff} idx={min_idx}")
        if min_idx > 0:
            self.plan_path = self.plan_path[min_idx + 1:]
            self.plan_qtrs = self.plan_qtrs[min_idx + 1:]
        

    def plan_a_star(self, ch_pts):
        if not self.do_need_new_plan(ch_pts):
            self.walk_path()
        else:
            print(f"new plan {self.start} => {self.goal}")
            if cfg.use_real_heigth_tolerances:
               tolerances = \
                   (self.cntr[2] + int(round(cfg.path_heigth_neg_real_tol / cfg.map_resolution)),\
                   self.cntr[2] + int(round(cfg.path_heigth_pos_real_tol / cfg.map_resolution)))
            else:
               tolerances = (self.cntr[2] + cfg.path_heigth_neg_vox_tol,\
                   self.cntr[2] + cfg.path_heigth_pos_vox_tol)


            self.replan_cnt += 1
            self.plan_path, self.plan_unf = a_star_3d(self.vox, self.pos, self.goal, tolerances)
            self.fat_path = make_fat_path(self.plan_path)
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
        self.replan_cnt += 1
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




    
