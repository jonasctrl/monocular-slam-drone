cmake_minimum_required(VERSION 3.0.2)
project(depth_estimation_package)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

# Add message files
add_message_files(
  FILES
  RGBWithPose.msg
  DepthWithPose.msg
  Pcd2WithPose.msg
)

# Generate added messages
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
