#!/usr/bin/env python3

import sys, os
import numpy as np
import time

from numba import njit
from scipy.spatial.transform import Rotation as R



import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Header
from nav.msg import Pcd2WithPose
from cv_bridge import CvBridge

from drrt import DRRT
from a_star import a_star_3d
from mapper import VoxArray 
import nav_config as cfg
from utils import bresenham3d_raycast, clamp


##########
#  MISC  #
##########

@njit
def in_bounds(voxels, x, y, z):
    s=voxels.shape
    return 0 <= x < s[0] and 0 <= y < s[1] and 0 <= z < s[2] 


@njit
def clamp(v, a, b):
    if v < a:
        return a
    if v > b:
        return b
    return v


@njit
def bresenham3d_raycast(p1, p2, voxels):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Compute deltas
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    
    # Determine the direction of the steps
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1
    sz = 1 if z2 > z1 else -1

    path = []
    
    # Initialize error terms
    if dx >= dy and dx >= dz:  # X is dominant
        err1 = 2 * dy - dx
        err2 = 2 * dz - dx
        while x1 != x2 and in_bounds(voxels, x1, y1, z1):
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            # voxels[x1, y1, z1] = empty_val
            if err1 > 0:
                y1 += sy
                err1 -= 2 * dx
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dx
            err1 += 2 * dy
            err2 += 2 * dz
            x1 += sx
    elif dy >= dx and dy >= dz:  # Y is dominant
        err1 = 2 * dx - dy
        err2 = 2 * dz - dy
        while y1 != y2 and in_bounds(voxels, x1, y1, z1): 
            # path.append((x1, y1, z1))
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            # voxels[x1, y1, z1] = empty_val
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dy
            if err2 > 0:
                z1 += sz
                err2 -= 2 * dy
            err1 += 2 * dx
            err2 += 2 * dz
            y1 += sy
    else:  # Z is dominant
        err1 = 2 * dx - dz
        err2 = 2 * dy - dz
        while z1 != z2 and in_bounds(voxels, x1, y1, z1):
            # path.append((x1, y1, z1))
            path.append((x1, y1, z1, voxels[x1, y1, z1]))
            # voxels[x1, y1, z1] = empty_val
            if err1 > 0:
                x1 += sx
                err1 -= 2 * dz
            if err2 > 0:
                y1 += sy
                err2 -= 2 * dz
            err1 += 2 * dx
            err2 += 2 * dy
            z1 += sz


    # Add the final voxel (end point)
    if in_bounds(voxels, x1, y1, z1):
        path.append((x2, y2, z2, voxels[x2, y2, z2]))

    return path

def depth_img_to_pcd(img, skip, factor, cam_params=None, fov=None, max_depth=float("inf")):
    height, width = img.shape
    point_cloud = []
    if cam_params is not None:
        (fx, fy, cx, cy) = cam_params
    elif fov is not None:
        fx = width / (2*tan(fov*pi/360))
        fy = fx * height / width
        cx = width / 2
        cy = height / 2
    else:
        raise Exception("'cam_params' or 'fov' must be specified ")

    for v in range(1, height, skip):
        for u in range(1, width, skip):
            z = img[v, u] / factor  # Depth value (in meters or millimeters)
            # if z == 0:  # Skip pixels with no depth
            if z == 0 or z > max_depth:  # Skip pixels with no depth
                continue

            # Convert (u, v, z) to (X, Y, Z)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            point_cloud.append([x, y, z])

    return point_cloud

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
        # vox[x, y, z] = cfg.occup_thr
        # vox[x, y, z] = cfg.ray_hit_incr
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
                # print(f"> {x} {y} {z} : {v} => {ch_val}")
                vox[x, y, z] = ch_val
                if v < cfg.occup_thr and ch_val >= cfg.occup_thr:
                    changed_pts.append((x, y, z, -1))
                elif v >= cfg.occup_thr and ch_val < cfg.occup_thr:
                    changed_pts.append((x, y, z, 0))
                # else:
                    # not_ch += 1

    # print(f"c={len(changed_pts)} nc={not_ch}")
    return changed_pts

def quaternion_from_two_vectors(v1, v2):
    # Normalize both vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    # Calculate the axis of rotation (cross product)
    axis = np.cross(v1, v2)
    
    # Calculate the angle between the two vectors (dot product and arccos)
    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    # If the vectors are parallel (angle is 0), return an identity quaternion
    if np.isclose(angle, 0):
        return R.from_quat([0, 0, 0, 1]).as_quat()  # No rotation

    # If the vectors are opposite, return a 180-degree rotation quaternion
    if np.isclose(angle, np.pi):
        # Find an arbitrary orthogonal axis
        orthogonal_axis = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
        axis = np.cross(v1, orthogonal_axis)
        axis = axis / np.linalg.norm(axis)
        return R.from_rotvec(np.pi * axis).as_quat()

    # Normalize the axis
    axis = axis / np.linalg.norm(axis)

    # Create quaternion from the axis-angle representation
    quaternion = R.from_rotvec(angle * axis)

    return quaternion.as_quat()

#################
#  MAPPER PART  #
#################


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

        travel_off = np.array(cfg.travel_off)
        self.goal:tuple = tuple((np.array(self.start) + travel_off).astype(int))

        self.pos:tuple = self.start
        self.pos_acc:tuple = self.start
        self.qtr = [0., 0., 0., 1.]

        self.data = []
        self.data_idx = 0

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
        self.goal = tuple(new_goal)
        self.updated_start_or_goal = True
        if update_start:
            self.start = self.pos 


    def get_plan(self, orig_coord=True):
        if len(self.plan_path) == 0:
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
        if not self.has_init_off:
            self.has_init_off = True
            self.init_off = np.array([cam_pos]) / self.res
        
        
        # Convert camera point to point in local voxel map
        # (tx, ty, tz) = cam_pos
        # cam_pos = (-ty, tx, tz)
        # cam_pos = (ty, tx, tz)
        c = tuple(self.point_to_map(np.array([cam_pos]))[0])
        c_acc = tuple(self.point_to_map_acc(np.array([cam_pos]))[0])

        # pcd = [(y, x, -z) for (x, y, z) in pcd]

        # Add position and quaterion to camera positions and orientations
        self.cam_qtrs.append(cam_qtr)
        self.cam_path.append(c)
        self.cam_path_acc.append(c)

        self.pos = c
        self.pos_acc = c_acc
        self.qtr = cam_qtr
 
        if not is_glob_fame:
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
        
        
    def __is_on_path(self, c):
        # print(f"is {c} in this path:{self.plan_path}")
        for p in self.plan_path:
            if p == c:
                # print(f"{c} is on the path")
                return True

        return False


    def __do_obst_interfere(self, path, obst):
        new_obst = [(x, y, z) for x, y, z, v in obst if v == -1]
        # del_obst = [(x, y, z) for x, y, z, v in obst if v == 0]

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
        
        
    def plan_a_star(self, ch_pts):
        if not self.updated_start_or_goal:
            if self.__is_on_path(self.pos):
                if not self.__do_obst_interfere(self.plan_path, ch_pts):
                    return 

        print(f"start={self.start} pos={self.pos} goal={self.goal}")
        # print(f"new plan {self.start} => {self.goal}")

        # print(f"start={tuple(self.point_from_map(self.start)[0])} pos={tuple(self.point_from_map(self.pos)[0])}")
        # print(f"new plan {tuple(self.point_from_map(self.start)[0])} => {tuple(self.point_from_map(self.goal)[0])}")

        self.plan_path = a_star_3d(self.vox, self.start, self.goal)
        self.updated_start_or_goal = False
        print(f"found path={self.plan_path}")
        self.__set_plan_qtrs()
        # print(f"found plan qtrs={self.plan_qtrs}")
        
        

    def plan_drrt(self, ch_pts):
        print(f"pos={self.pos}")
        self.pf.update_obstacles(ch_pts)
        self.pf.update_start(self.pos)
        # print("replaning")
        self.pf.plan()
        self.pf.plan(force_iters=50)
        # print("getting path")
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



##############
#  ROS PART  #
##############

def pointcloud2_to_array(msg):
    # Extract the fields from PointCloud2 message
    points = []
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([point[0], point[1], point[2]])
    return np.array(points)

class MapperNavNode:
    def __init__(self):
        self.vmap = VoxArray(resolution=cfg.map_resolution, shape=[600,600,300])
        rospy.init_node('mapper_nav', anonymous=True)

        self.bridge = CvBridge()
        self.occupied_pub = rospy.Publisher('/occupied_space', PointCloud2, queue_size=1)
        self.empty_pub = rospy.Publisher('/empty_space', PointCloud2, queue_size=1)
        self.cam_path_pub = rospy.Publisher('/cam_path', Path, queue_size=1)
        self.plan_path_pub = rospy.Publisher('/plan_path', Path, queue_size=1)
        self.plan_map_path_pub = rospy.Publisher('/plan_map_path', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/map_pose', PoseStamped, queue_size=1)

        self.start_pub = rospy.Publisher('/nav_start', PointStamped, queue_size=1)
        self.goal_pub = rospy.Publisher('/nav_goal', PointStamped, queue_size=1)

        # self.depth_sub = rospy.Subscriber('/ground_truth/depth_with_pose', DepthWithPose, self.image_callback)
        self.depth_sub = rospy.Subscriber('/cam_pcd_pose', Pcd2WithPose, self.pcd_pose_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)

        rospy.loginfo("Mapper-Navigation node initialized.")


    def publish_cam_path_msg(self):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for pt, qtr in zip(self.vmap.cam_path_acc, self.vmap.cam_qtrs):
            (tx, ty, tz) = pt
            (qx, qy, qz, qw) = qtr

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = rospy.Time.now()

            pose_stamped.pose.position.x = tx
            pose_stamped.pose.position.y = ty
            pose_stamped.pose.position.z = tz

            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw

            path_msg.poses.append(pose_stamped)

        self.cam_path_pub.publish(path_msg)

    def publish_plan_path_msg(self, orig=True):
        path_msg = Path()

        path_msg.header = Header()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        path, qtrs = self.vmap.get_plan(orig_coord=orig)
        for pt, qtr in zip(path, qtrs):
            (tx, ty, tz) = pt
            (qx, qy, qz, qw) = qtr
            
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "map"
            pose_stamped.header.stamp = rospy.Time.now()

            pose_stamped.pose.position.x = tx
            pose_stamped.pose.position.y = ty
            pose_stamped.pose.position.z = tz

            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw

            path_msg.poses.append(pose_stamped)

        if orig:
            self.plan_path_pub.publish(path_msg)
        else:
            self.plan_map_path_pub.publish(path_msg)

    def publish_map_pose_msg(self):
        (pt, qtr) = self.vmap.get_pose(orig_coord=False)
        (tx, ty, tz) = pt
        (qx, qy, qz, qw) = qtr

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()

        pose_stamped.pose.position.x = tx
        pose_stamped.pose.position.y = ty
        pose_stamped.pose.position.z = tz

        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw

        self.pose_pub.publish(pose_stamped)


    def publish_start_msg(self):
        (x, y, z) = self.vmap.get_start(orig_coord=False)
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"  # Adjust to your frame of reference
        point_msg.point = Point(x, y, z)
        self.start_pub.publish(point_msg)

    def publish_goal_msg(self):
        (x, y, z) = self.vmap.get_goal(orig_coord=False)
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map"  # Adjust to your frame of reference
        point_msg.point = Point(x, y, z)
        self.goal_pub.publish(point_msg)

    def publish_occupied_space_msg(self):
        points = self.vmap.get_occupied_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.occupied_pub.publish(pcd_msg)

    def publish_empty_space_msg(self):
        points = self.vmap.get_empty_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.empty_pub.publish(pcd_msg)

    def goal_callback(self, msg):
        position = msg.pose.position
        pos = tuple(np.array([position.x, position.y, 77]).astype(int))
        print(f"Received goal={pos}")
        self.vmap.set_goal(pos, update_start=True)


    def pcd_pose_callback(self, msg):
        
        t0 = time.time()
        
        pcd = pointcloud2_to_array(msg.pcd)

        pos_pt = msg.position
        pos = [pos_pt.x, pos_pt.y, pos_pt.z]

        qtr_pt = msg.orientation
        qtr = [qtr_pt.x, qtr_pt.y, qtr_pt.z, qtr_pt.w]

        is_glob_fame = msg.is_global_frame.data

        t1 = time.time()
        ch_pts = self.vmap.update(pcd, pos, qtr, is_glob_fame)
        
        t2 = time.time()

        
        if cfg.publish_occup:
            self.publish_occupied_space_msg()

        if cfg.publish_empty:
            self.publish_empty_space_msg()
        
        t3 = time.time()
        self.vmap.plan(ch_pts)
        t4 = time.time()


        if cfg.publish_pose:
            self.publish_map_pose_msg()


        if cfg.publish_path:
            self.publish_cam_path_msg()

        if cfg.publish_plan:
            self.publish_start_msg()
            self.publish_goal_msg()
            self.publish_plan_path_msg(orig=False) # Map scale
            self.publish_plan_path_msg(orig=True) # Original scale

        t5 = time.time()

        print(f"init:{round(t1-t0, 4)} mapping:{round(t2-t1, 4)} pub1:{round(t3-t2, 4)} plan:{round(t4-t3, 4)} pub2:{round(t5-t4, 4)}")


    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down depth estimation node...")


if __name__ == '__main__':
    try:
        node = MapperNavNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

