
#############
#  mapping  #
#############
map_depth = 600
map_width = 600
map_heigth = 200

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


# map_resolution = 0.5
# map_resolution = 1
# map_resolution = 1.5
map_resolution = 2
# map_resolution = 2.89
# map_resolution = 3
# map_resolution = 4
# map_resolution = 8


######################
#  data proccessing  #
######################

dimg_stride = 2
dimg_min_depth = 1
dimg_max_depth = 50

###############
#  maze part  #
###############


cam_w = 18
cam_h = 12
cam_depth = 9
cam_scaling = 8



##################
#  path finding  #
##################

# travel_off = (20, 5, 2) 
# travel_off = (0, 0, 10) 
# travel_off = (0, 5, 0) 
travel_off = (0, 0, 0) 

use_drrt = False
use_a_star = True

path_tolerance = 3.0


################
#  publishing  #
################

# publish_occup = False
publish_occup = True
publish_empty = False

publish_pose = True
publish_path = True
publish_plan = True


