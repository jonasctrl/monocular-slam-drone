
import argparse

#############
#  mapping  #
#############

# Map depth
map_depth = 600
# Map width
map_width = 600
# Map height
map_heigth = 100

# "Unknown" voxel occupancy value
occup_unkn = -128
# Minimum known voxel occupancy value
occup_min = 0
# Maximum known voxel occupancy value
occup_max = 12
# Threshold from which voxel is considered "occupied"
occup_thr = 8

# Constant by which voxel occupancy value is decreased if ray passes thorugh it
ray_miss_incr = -1

# Constant by which voxel occupancy value is increased if obstacle is detected in it
ray_hit_incr = 8

# Mapping resolution, i.e. amount of simulation distance units (m, ft, cm, etc.) for each voxel
map_resolution = 1

######################
#  data proccessing  #
######################

# Use opencv feature matching
use_opencv_imaging = False

# Use rgb-to-depth imaging
use_rgb_imaging = False

# Stride at which depth image is downsampled
dimg_stride = 2

# Minimum distance to point ((in simulation distance units)
dimg_min_depth = 1
# Maximum distance to point (in simulation distance units)
dimg_max_depth = 50

##################
#  path finding  #
##################

# Goal position offset from start position in simulation units
goal_off = ()

# Goal position offset from start position in voxels
goal_off_vox = ()

# Use DRRT path finding algorithm instead of A*
use_drrt = False
# Use A* path finding algorithm instead of DRRT
use_a_star = True

# Maximum allowed A* iterations
max_a_star_iters = 500

# How much to allowed A* iterations if path finding do not get finished
unf_max_iters_incr = 200

# How many voxels viecle is allowed to drift from path without triggering replanning
path_drift_tolerance = 3.0

# Upper path finding heigth restriction as offset in voxels from starting height
path_heigth_pos_vox_tol = 2
# Lower path finding heigth restriction as offset in voxels from starting height
path_heigth_neg_vox_tol = -2

# Upper path finding heigth restriction as offset in simulation distance units from starting height
path_heigth_pos_real_tol = 3
# Lower path finding heigth restriction as offset in simulation distance units from starting height
path_heigth_neg_real_tol = -3

# How much to decrease lower path finding heigth restriction value (in voxels)
unf_neg_tol_incr = 0

# How much to increase higher path finding heigth restriction value (in voxels)
unf_pos_tol_incr = 0

# Use simulation unit based heigth restrictions instead of voxel based ones
use_real_heigth_tolerances = True

# Minimum amount of "unfinished" plan steps allowed before triggering replanning
unf_plan_limit = 5

################
#  publishing  #
################

# Publish intensities in "occupied space" ROS message
publish_occup_intensities = True

# Publish "occupied space" ROS message
publish_occup = True

# Publish "empty known space" ROS message
publish_empty = False

# Publish "viecle pose" ROS message
publish_pose = True

# Publish "viecle path" ROS message
publish_path = True

# Publish "viecle planned path" ROS message
publish_plan = True

################
#  simulation  #
################

# Reset simulation on startup
reset_sim = False

# Exit if goal is reached
exit_on_goal = False

# Simulation camera name
camera_name = "fc"

# Maximum frequency of simulation
max_sim_freq= 20

# Maximum speed of viecle in simulation
speed = 2.0

#############
#  logging  #
#############

# File for results logging
logfile = None

# Number of failures to find path before quiting simulation
no_path_watchdog_cnt = 50

# Number of collision iteration before quiting simulation
collision_watchdog_cnt = 50

# Number of allowed iterations in simulation
max_steps = 100000

# Allowed distance error to goal in simulation distance units to to caount as 'reached'
goal_err = 5

globals_dict = {
    key: value
    for key, value in globals().items()
    # if not key.startswith("__") and not callable(value)
    if not key.startswith("__") and not callable(value) and not isinstance(value, type(argparse))
}

def parse_arguments():
    global map_depth, map_width, map_heigth, occup_unkn, occup_min, occup_max
    global occup_thr, ray_miss_incr, ray_hit_incr, map_resolution
    global use_opencv_imaging, use_rgb_imaging, dimg_stride
    global dimg_min_depth, dimg_max_depth, goal_off, goal_off_vox, use_drrt, use_a_star
    global max_a_star_iters, path_drift_tolerance, path_heigth_pos_vox_tol
    global path_heigth_neg_vox_tol, path_heigth_pos_real_tol, path_heigth_neg_real_tol
    global use_real_heigth_tolerances, unf_plan_limit, publish_occup_intensities
    global publish_occup, publish_empty, publish_pose, publish_path, publish_plan
    global reset_sim, camera_name, max_sim_freq, speed, logfile, exit_on_goal
    global unf_max_iters_incr, unf_neg_tol_incr, unf_pos_tol_incr, no_path_watchdog_cnt
    global collision_watchdog_cnt, goal_err, max_steps

    parser = argparse.ArgumentParser(description="Simulation configuration options.")

    # Mapping options
    parser.add_argument('--map_depth', type=int, default=map_depth, help="Map depth.")
    parser.add_argument('--map_width', type=int, default=map_width, help="Map width.")
    parser.add_argument('--map_heigth', type=int, default=map_heigth, help="Map height.")
    parser.add_argument('--occup_unkn', type=int, default=occup_unkn, help="Unknown voxel occupancy value.")
    parser.add_argument('--occup_min', type=int, default=occup_min, help="Minimum known voxel occupancy value.")
    parser.add_argument('--occup_max', type=int, default=occup_max, help="Maximum known voxel occupancy value.")
    parser.add_argument('--occup_thr', type=int, default=occup_thr, help="Threshold for occupied voxels.")
    parser.add_argument('--ray_miss_incr', type=int, default=ray_miss_incr, help="Ray miss increment.")
    parser.add_argument('--ray_hit_incr', type=int, default=ray_hit_incr, help="Ray hit increment.")
    parser.add_argument('--map_resolution', type=float, default=map_resolution, help="Mapping resolution.")

    # Data processing options
    parser.add_argument('--use_opencv_imaging', action='store_true', default=use_opencv_imaging, help="Enable OpenCV imaging.")
    parser.add_argument('--use_rgb_imaging', action='store_true', default=use_rgb_imaging, help="Enable RGB imaging.")
    parser.add_argument('--dimg_stride', type=int, default=dimg_stride, help="Depth image downsampling stride.")
    parser.add_argument('--dimg_min_depth', type=int, default=dimg_min_depth, help="Minimum depth.")
    parser.add_argument('--dimg_max_depth', type=int, default=dimg_max_depth, help="Maximum depth.")

    # Pathfinding options
    parser.add_argument('--goal_off', nargs='+', type=float, default=goal_off, help="Goal position offset in simulation units.")
    parser.add_argument('--goal_off_vox', nargs='+', type=float, default=goal_off_vox, help="Goal position offset in voxels.")
    parser.add_argument('--use_drrt', action='store_true', default=use_drrt, help="Use DRRT algorithm.")
    parser.add_argument('--use_a_star', action='store_true', default=use_a_star, help="Use A* algorithm.")
    parser.add_argument('--max_a_star_iters', type=int, default=max_a_star_iters, help="Maximum A* iterations.")
    parser.add_argument('--unf_max_iters_incr', type=int, default=unf_max_iters_incr, help="How much to allowed A* iterations if path finding do not get finished.")
    parser.add_argument('--path_drift_tolerance', type=float, default=path_drift_tolerance, help="Path drift tolerance.")
    parser.add_argument('--path_heigth_pos_vox_tol', type=int, default=path_heigth_pos_vox_tol, help="Upper height restriction (voxels).")
    parser.add_argument('--path_heigth_neg_vox_tol', type=int, default=path_heigth_neg_vox_tol, help="Lower height restriction (voxels).")
    parser.add_argument('--path_heigth_pos_real_tol', type=float, default=path_heigth_pos_real_tol, help="Upper height restriction (real units).")
    parser.add_argument('--path_heigth_neg_real_tol', type=float, default=path_heigth_neg_real_tol, help="Lower height restriction (real units).")

    parser.add_argument('--unf_neg_tol_incr', type=int, default=unf_neg_tol_incr, help="How much to decrease lower path finding heigth restriction value (in voxels).")
    parser.add_argument('--unf_pos_tol_incr', type=int, default=unf_pos_tol_incr, help="How much to increase higher path finding heigth restriction value (in voxels).")
    parser.add_argument('--not_use_real_heigth_tolerances', action='store_false', default=use_real_heigth_tolerances, help="Use real unit height restrictions.")
    parser.add_argument('--unf_plan_limit', type=int, default=unf_plan_limit, help="Minimum unfinished steps to replan.")

    # Publishing options
    parser.add_argument('--not_publish_occup_intensities', action='store_false', default=publish_occup_intensities, help="Publish occupancy intensities.")
    parser.add_argument('--publish_occup', action='store_true', default=publish_occup, help="Publish occupied space.")
    parser.add_argument('--publish_empty', action='store_true', default=publish_empty, help="Publish empty space.")
    parser.add_argument('--not_publish_pose', action='store_false', default=publish_pose, help="Publish vehicle pose.")
    parser.add_argument('--not_publish_path', action='store_false', default=publish_path, help="Publish vehicle path.")
    parser.add_argument('--not_publish_plan', action='store_false', default=publish_plan, help="Publish planned path.")

    # Simulation options
    parser.add_argument('--reset_sim', action='store_true', default=reset_sim, help="Reset simulation on startup.")
    parser.add_argument('--exit_on_goal', action='store_true', default=exit_on_goal, help="Exit if goal is reached.")
    parser.add_argument('--camera_name', type=str, default=camera_name, help="Simulation camera name.")
    parser.add_argument('--max_sim_freq', type=int, default=max_sim_freq, help="Maximum simulation frequency.")
    parser.add_argument('--speed', type=float, default=speed, help="Maximum vehicle speed.")

    # Logging
    parser.add_argument('--logfile', type=str, default=logfile, help="File for results logging.")
    parser.add_argument('--goal_err', type=int, default=goal_err, help="Allowed distance error to goal in simulation distance units to to caount as 'reached'.")

    parser.add_argument('--collision_watchdog_cnt', type=int, default=collision_watchdog_cnt, help="Number of collision iteration before quiting simulation.")

    parser.add_argument('--no_path_watchdog_cnt', type=int, default=no_path_watchdog_cnt, help="Number of failures to find path before quiting simulation.")
    
    parser.add_argument('--max_steps', type=int, default=max_steps, help="Number of allowed iterations in simulation.")

    # Parse arguments
    args = parser.parse_args()

    # Update globals with parsed arguments
    globals().update(vars(args))


def get_configs():
    gdict = {
        key: value
        for key, value in globals().items()
        if globals_dict.get(key) is not None
    }

    return gdict
