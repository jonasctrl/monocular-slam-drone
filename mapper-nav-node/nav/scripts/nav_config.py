
#############
#  mapping  #
#############
map_depth = 600
map_width = 600
map_heigth = 100

# map_depth = 300
# map_width = 300
# map_heigth = 100

occup_unkn = -128
occup_min = 0
# occup_max = 12
occup_max = 12
occup_thr = 8
# occup_thr = 5
occup_fade = -1

ray_miss_incr = -1
ray_hit_incr = 7
# ray_hit_incr = 9


# map_resolution = 0.5
# map_resolution = 1
map_resolution = 1.5
# map_resolution = 2
# map_resolution = 2.89
# map_resolution = 3
# map_resolution = 4
# map_resolution = 8


######################
#  data proccessing  #
######################

# use_opencv_imaging = True
use_opencv_imaging = False

# use_rgb_imaging = True
use_rgb_imaging = False

dimg_stride = 2
dimg_min_depth = 1
dimg_max_depth = 50



##################
#  path finding  #
##################

# travel_off = (20, 5, 2) 
# travel_off = (0, 0, 10) 
# travel_off = (0, 5, 0) 
travel_off = (0, 0, 0) 

max_a_star_iters = 500

use_drrt = False
use_a_star = True

path_drift_tolerance = 3.0

path_heigth_pos_vox_tol = 2
path_heigth_neg_vox_tol = -2

path_heigth_pos_real_tol = 3
path_heigth_neg_real_tol = -3

use_real_heigth_tolerances = True
# use_real_heigth_tolerances = False

unf_plan_limit = 4


################
#  publishing  #
################

publish_occup_intensities = True
# publish_occup_intensities = False

# publish_occup = False
publish_occup = True
publish_empty = False

publish_pose = True
publish_path = True
publish_plan = True


