import numpy as np
import random

# import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from a_star import a_star_3d

from utils import bresenham3d_get_collision
# from utils import bresenham3d_check_known_space, depth_img_to_pcd

cam_w = 12
cam_h = 8
cam_depth = 6
cam_scaling = 12

def_cam_target_pts = np.array([(cam_depth, w - cam_w / 2, h - cam_h / 2) for w in range(cam_w) for h in range(cam_h)])
def_cam_pos = [0,0,0]

class Maze(object):
    def __init__(self, edge=300, height=50, num_obstacles=30):
        self.env_map = self.generate_voxel_map_with_obstacles((edge, edge, height), num_obstacles=num_obstacles)
        self.cur_pos = [10,50,5]
        # self.cur_pos = [0,0,0]
        self.cur_qtr = [0,0,0,1]
        self.pos_history = [self.cur_pos]
        self.qtr_history = [self.cur_qtr]
        self.pos_plan = []
        self.qtr_plan = []
        self.glob_pcd = []
        self.cam_pcd = []
        self.cam_targets = []

        self.bounds = [(0, edge), (0, edge), (0, height)]


        # goal = (200, 200, 10)
        # goal = (60,50,20)
        # goal = (40,60,4)
        goal = (100,80,4)
        start = tuple(self.cur_pos)
        print(f"searching path from {start} to {goal}")
        nav_path = a_star_3d(self.env_map, start, goal)
        nav_qtrs = len(nav_path) * [[0,0,0,1]]
        print(f"found path={nav_path}")
        print(f"nav_qtrs={nav_qtrs}")
        self.set_plan(nav_path, nav_qtrs)


        self.step()
        
    def generate_voxel_map_with_obstacles(self, shape, num_obstacles, seed=15):
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
        # voxel_map[0,:,:] = 1
        # voxel_map[x_size-1,:,:] = 1
        # voxel_map[:,0,:] = 1
        # voxel_map[:,y_size-1,:] = 1
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
        pos = np.array(self.cur_pos)
        
        cam_targets = def_cam_target_pts.copy()
        cam_targets = cam_scaling * cam_targets
        cam_targets = qtr.apply(cam_targets)
        cam_targets = pos + cam_targets
        cam_targets = cam_targets.astype(int)

        collided = []
        
        # print(f"cam_targets:{cam_targets}")
        for ctg in cam_targets:
            col = bresenham3d_get_collision(pos, ctg, self.env_map)
            if col is not None:
                collided.append(col)

        self.cam_targets = cam_targets
        self.glob_pcd = np.array(collided)
        # self.glob_pcd = cam_targets
        
        cam_pcd = np.array(collided)
        cam_pcd = cam_pcd - pos
        cam_pcd = inv_qtr.apply(cam_pcd)

        self.cam_pcd = cam_pcd
        # self.cam_pcd = cam_targets

    def set_plan(self, new_pos_plan:list, new_qtr_plan:list):
        self.pos_plan = new_pos_plan
        self.qtr_plan = new_qtr_plan

    # def navigate(self, start, goal):
        # nav_path = a_star_3d(self.vox, start, goal)
        # print(f"found path={nav_path}")

    def step(self):
        if len(self.pos_plan) == 0:
            print(f"No more planned steps")
            # return 

        if len(self.pos_plan):
            self.cur_pos = self.pos_plan.pop(0)
            print(f"Steping to {self.cur_pos}")

        if len(self.qtr_plan):
            self.cur_qtr = self.qtr_plan.pop(0)

        self.pos_history.append(self.cur_pos) 
        self.qtr_history.append(self.cur_qtr) 

        self.cam_collisions()



    # def visualize_voxel_map(self, cam_targets=None, cam_coll=None, camera=None):
        # x, y, z = np.where(self.env_map == 1)  # Get the coordinates of obstacles
        
        # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', 
                                           # marker=dict(size=2, color=z, colorscale='viridis'))])
        
        # if cam_targets is not None and len(cam_targets) > 0:
            # x = cam_targets[:,0]
            # y = cam_targets[:,1]
            # z = cam_targets[:,2]
            # fig = fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                               # marker=dict(size=2, color="blue"), name='cam_target'))

        # if cam_coll is not None and len(cam_coll) > 0:
            # x = cam_coll[:,0]
            # y = cam_coll[:,1]
            # z = cam_coll[:,2]
            # fig = fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                               # marker=dict(size=2, color="red"), name='cam_collisions'))
        # if camera is not None:
            # x = [camera[0]]
            # y = [camera[1]]
            # z = [camera[2]]
            # fig = fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                               # marker=dict(size=2, color="green"), name='camera'))
        # fig.update_layout(scene=dict(aspectmode='data'), title="3D Voxel Map")
        # fig.show()


# if __name__ == "__main__":
    # grid_st = np.array([0, 0, 0]).astype(int)
    # grid_shape = np.array([600, 600, 200]).astype(int)
    # center = np.array([grid_shape[0]//2, grid_shape[1]//2, 50]).astype(int)
    # resolution = 2.5

    # cam_qtrs = generate_z_axis_quaternions(12)

    # env_map = generate_voxel_map_with_obstacles(edge, edge, height, num_obstacles=30)
    # vmap = VoxArray(center, resolution, grid_shape, grid_st)
    
    
    # path_len = 12
    
    # cam_pos = [218,175,40]
    # cam_poses = path_len * cam_pos
    # # cam_qtr = [0,0,0,1]
    # cam_qtr = [0.939,-0.052,-0.296,0.116]

    # cam_tgs = []
    # cam_col = []
    # for i in range(path_len):
        # ctg, col = get_cam_collisions(cam_pos, cam_qtr, env_map)
        # cam_tgs.append(ctg)
        # cam_col.append(col)

    # # cam_tgs, cam_coll = get_cam_collisions(cam_pos, cam_qtr, env_map)
    # # data_arr = parse_airsim_data_v3(img_end, img_skip)
    # # data_arr = parse_airsim_data(img_end, img_skip)
    # # data_arr = parse_data(img_end, img_skip)
    # # for i in range(len(data_arr)):
        # # vmap.add_pcd_from_datapoint(data_arr[i])        
        
    # # vmap.navigate(start, goal)
    # # vmap.plot()
    # # vmap.plot(use_confidence=True)
    
    # visualize_voxel_map(env_map, cam_targets=cam_tgs, cam_coll=cam_col, camera=cam_pos)



