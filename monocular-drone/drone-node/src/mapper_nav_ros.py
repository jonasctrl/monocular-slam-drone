#!/usr/bin/env python3

import math
import struct
import time
import numpy as np
import pyfiglet

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point
from std_msgs.msg import Header

from monodepth import MonoDepth2DepthEstimatorModule
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import airsim

from mapper import VoxArray, precompile
from data_logger import DataLogger
import nav_config as cfg

import warnings
warnings.filterwarnings("ignore")

class MapperNavNode:
    def __init__(self):
        self.sequence = 0
        self.vmap = VoxArray(resolution=cfg.map_resolution, shape=(cfg.map_width,cfg.map_depth,cfg.map_heigth))
        rospy.init_node('mapper_nav', anonymous=True)

        self.depth_estimator_module = MonoDepth2DepthEstimatorModule()
        
        self.client = airsim.MultirotorClient(ip="host.docker.internal", port=41451)
        self.client.confirmConnection()

        print(f"Reseting simulation ... ", end="")
        time.sleep(1)
        
        self.client.reset()
        time.sleep(1)
        
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        print(f"Reseting simulation ... ", end="")

        self.ctl = DroneController(self.client)
        self.logger = DataLogger()


        self.occupied_pub = rospy.Publisher('/occupied_space', PointCloud2, queue_size=1)
        self.empty_pub = rospy.Publisher('/empty_space', PointCloud2, queue_size=1)
        self.cam_path_pub = rospy.Publisher('/cam_path', Path, queue_size=1)
        self.plan_path_pub = rospy.Publisher('/plan_path', Path, queue_size=1)
        self.plan_map_path_pub = rospy.Publisher('/plan_map_path', Path, queue_size=1)
        self.pose_pub = rospy.Publisher('/map_pose', PoseStamped, queue_size=1)

        self.start_pub = rospy.Publisher('/nav_start', PointStamped, queue_size=1)
        self.goal_pub = rospy.Publisher('/nav_goal', PointStamped, queue_size=1)

        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)

        rospy.loginfo("Mapper-Navigation initialized.")

    ################
    #  PUBLISHERS  #
    ################

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
        point_msg.header.frame_id = "map"  
        point_msg.point = Point(x, y, z)
        self.start_pub.publish(point_msg)

    def publish_goal_msg(self):
        (x, y, z) = self.vmap.get_goal(orig_coord=False)
        point_msg = PointStamped()
        point_msg.header.stamp = rospy.Time.now()
        point_msg.header.frame_id = "map" 
        point_msg.point = Point(x, y, z)
        self.goal_pub.publish(point_msg)

    def publish_occupied_space_msg(self):
        points, intensities = self.vmap.get_occupied_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        if cfg.publish_occup_intensities:
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            # Pack data into binary format
            data = []
            for point, intensity in zip(points, intensities):
                data.append(struct.pack('ffff', point[0], point[1], point[2], intensity))

            data = b''.join(data)

            pcd_msg = PointCloud2(
                header=header,
                height=1,
                width=len(points),
                fields=fields,
                is_bigendian=False,
                # 4 fields * 4 bytes/field
                point_step=16,  
                row_step=16 * len(points),
                data=data,
                is_dense=True
            )
        else:
            pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.occupied_pub.publish(pcd_msg)

    def publish_empty_space_msg(self):
        points = self.vmap.get_empty_space_pcd()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"
        pcd_msg = pc2.create_cloud_xyz32(header, points)

        self.empty_pub.publish(pcd_msg)

    ################
    #  SUBSCRIBERS #
    ################

    def goal_callback(self, msg):
        position = msg.pose.position
        pos = tuple(np.array([position.x, position.y, self.vmap.cntr[2]]).astype(int))
        self.vmap.set_goal(pos, update_start=True)

    ######################
    #  DEPTH ESTIMATION  #
    ######################

    def get_rgb_img(self):
        response = self.client.simGetImages([
            airsim.ImageRequest(cfg.camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)
            ])[0]

        img_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
        return img_rgb

    def get_gray_img(self):
        img_rgb = self.get_rgb_img()
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return img_gray

    
    def estimate_depth(self):
        img_rgb = self.get_rgb_img()
        depth_map = self.depth_estimator_module.generate_depth_map(img_rgb)

        return depth_map
        
    
    def get_depth_img(self):
        response = self.client.simGetImages(
            [airsim.ImageRequest(cfg.camera_name, airsim.ImageType.DepthPlanar, pixels_as_float=True, compress=False)])[0]

        depth_map= np.array(response.image_data_float, dtype=np.float32).reshape(response.height, response.width)
        return depth_map
        

    def depth_img_to_pcd(self, dimg, fov):
        factor = 1.0
        (image_height, image_width) = dimg.shape

        # Compute fx and fy from FOV
        hfov_rad = fov * math.pi / 180.0
        fx = (image_width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx
        # Compute principal point coordinates
        cx = image_width / 2.0
        cy = image_height / 2.0

        # Compute 3D points from the depth map
        pcd = []
        for v in range(0, image_height, cfg.dimg_stride):
            for u in range(0, image_width, cfg.dimg_stride):
                # z = dimg[v, u]
                z = dimg[v, u] / factor
                if z >= cfg.dimg_min_depth and z < cfg.dimg_max_depth:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    # rotation 90 deg OX
                    pcd.append([z, -x, -y])

        return pcd
        
    def get_kinematics(self):
        kinematics = self.client.simGetGroundTruthKinematics()

        v = kinematics.linear_velocity  # Velocity as a Vector3r (x, y, z)
        a = kinematics.linear_acceleration  # Acceleration as a Vector3r (x, y, z)
        av = kinematics.angular_velocity  # Angular velocity (rad/s)
        aa = kinematics.angular_acceleration  # Angular acceleration (rad/s^2)

        vel = (v.x_val, -v.y_val, -v.z_val)
        acc = (a.x_val, -a.y_val, -a.z_val)
        avel = (av.x_val, -av.y_val, -av.z_val)
        aacc = (aa.x_val, -aa.y_val, -aa.z_val)


        return vel, acc, avel, aacc


    def get_camera_info(self):
        camera_info = self.client.simGetCameraInfo(cfg.camera_name)
        fov = camera_info.fov

        p = camera_info.pose.position
        # NED to ENU
        pt = np.array([p.x_val, -p.y_val, -p.z_val])
        o = camera_info.pose.orientation
        # NED to ENU
        ori = np.array([o.x_val, -o.y_val, -o.z_val, o.w_val])

        return fov, pt, ori

    def get_collisions(self):
        collision_info = self.client.simGetCollisionInfo()
        has_collided = collision_info.has_collided  # Boolean indicating if a collision occurred
        collision_object_id = collision_info.object_id  # ID of the object involved in the collision

        return has_collided, collision_object_id


    def step(self):
        t0 = time.time()
        self.logger.new_entry()
        self.logger.add_key_val("idx", self.sequence)
        fov, position, orientation = self.get_camera_info()
        self.logger.add_key_val("real_pos", tuple(position))
        self.logger.add_key_val("real_ori", tuple(orientation))
        
        v, a, av, aa = self.get_kinematics()
        self.logger.add_key_val("vel", v)
        self.logger.add_key_val("acc", a)
        self.logger.add_key_val("ang_vel", av)
        self.logger.add_key_val("ang_acc", aa)

        has_col, col_obj_id = self.get_collisions()
        self.logger.add_key_val("has_col", has_col)
        self.logger.add_key_val("col_obj_id", col_obj_id)

        if has_col:
            print(f"Vehicle is colliding")
            self.vmap.coll_cnt += 1
        else:
            self.vmap.coll_cnt = 0


        self.logger.add_key_val("time_start", time.time())
        if cfg.use_rgb_imaging:
            depth_data = self.estimate_depth()
        else:
            depth_data = self.get_depth_img()

        if depth_data is None:
            rospy.logwarn("No depth data")
        

        t1 = time.time()
        
        pcd = self.depth_img_to_pcd(depth_data, fov)
        
        self.logger.add_key_val("pcd_len", len(pcd))
        pos = position
        qtr = orientation

        t2 = time.time()

        ch_pts = self.vmap.update(pcd, pos, qtr, False)
        self.logger.add_key_val("ch_pts_len", len(ch_pts))
        
        t3 = time.time()
        self.vmap.plan(ch_pts)
        t4 = time.time()


        path, _ = self.vmap.get_plan(orig_coord=True)
        self.logger.add_key_val("plan_len", len(path))
        self.logger.add_key_val("replan_cnt", self.vmap.replan_cnt)
        cur_pos = self.vmap.get_pose(orig_coord=True)[0]
        self.logger.add_key_val("map_pos", tuple(cur_pos))
        self.logger.add_key_val("plan_pos",  tuple(path[0]) if len(path) > 0 else tuple(cur_pos))

        self.logger.add_key_val("time_end", time.time())
        
        
        goal_orig = self.vmap.point_from_map(self.vmap.goal)
        goal_err = np.linalg.norm(np.array(cur_pos) - np.array(goal_orig))
        if goal_err <= cfg.goal_err and (len(cfg.goal_off_vox) == 3 or len(cfg.goal_off) == 3):
            self.logger.add_key_val("gloal_reached", True)
            self.logger.push_entry()
            self.logger.export_logs()
            if cfg.exit_on_goal:
                print(f"Goal was reached. Exiting ...")
                self.client.landAsync().join()
                self.client.armDisarm(False)
                exit(0)
                
        self.logger.add_key_val("gloal_reached", False)
        self.logger.push_entry()

        if self.vmap.coll_cnt > cfg.collision_watchdog_cnt:
            print(f"Collision counter exdeeded. Exiting ...")
            self.logger.export_logs()
            self.client.landAsync().join()
            self.client.armDisarm(False)
            exit(0)
        
        if self.vmap.no_path_cnt > cfg.no_path_watchdog_cnt:
            print(f"No path counter exdeeded. Exiting ...")
            self.logger.export_logs()
            self.client.landAsync().join()
            self.client.armDisarm(False)
            exit(0)
                
        if self.sequence > cfg.max_steps:
            print(f"Maximum iteration counter exdeeded. Exiting ...")
            self.logger.export_logs()
            self.client.landAsync().join()
            self.client.armDisarm(False)
            exit(0)
        
        if len(path) > 1:
            self.ctl.move_along_path(path)


        t5 = time.time()
        
        if cfg.publish_occup:
            self.publish_occupied_space_msg()

        if cfg.publish_empty:
            self.publish_empty_space_msg()
        
        if cfg.publish_pose:
            self.publish_map_pose_msg()


        if cfg.publish_path:
            self.publish_cam_path_msg()

        if cfg.publish_plan:
            self.publish_start_msg()
            self.publish_goal_msg()
            self.publish_plan_path_msg(orig=False) # Map scale
            self.publish_plan_path_msg(orig=True) # Original scale

        t6 = time.time()

        print(f"img: {round(t1-t0, 3)} to_pcd:{round(t2-t1, 3)} map:{round(t3-t2, 3)} plan:{round(t4-t3, 3)} move:{round(t5-t4, 3)} pub:{round(t6-t5, 3)}")
            

    def run(self):
        rate = rospy.Rate(cfg.max_sim_freq)

        while not rospy.is_shutdown():
            self.step()

            self.sequence += 1
            rate.sleep()


##################
#  DRONE CONTROL #
##################

class DroneController:
    def __init__(self, client):
        self.client = client


    def move_along_path(self, path):
        path = [airsim.Vector3r(x, -y, -z) for (x, y, z) in path]
        self.client.moveOnPathAsync(
            path=path,
            velocity=cfg.speed,
            timeout_sec=10,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
            lookahead=-1,
            adaptive_lookahead=0
        )

def display_banner():
    banner = pyfiglet.figlet_format("MONOCULAR DRONE", font="slant")
    print(banner)

if __name__ == '__main__':
    node = None
    try:
        cfg.parse_arguments()

        display_banner()
        print(f"{cfg.get_configs()}")
        
        precompile()

        node = MapperNavNode()
        node.run()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected.")
        if node is not None:
            if node.logger is not None:
                node.logger.export_logs()
            if node.client is not None:
                node.client.landAsync().join()
                
                # NOTE: Disarm and disable API control
                # node.client.armDisarm(False)
                # node.client.enableApiControl(False)
    except rospy.ROSInterruptException:
        pass
