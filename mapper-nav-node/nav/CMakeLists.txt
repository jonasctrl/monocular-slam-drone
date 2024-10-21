cmake_minimum_required(VERSION 3.0.2)
project(nav)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  RGBWithPose.msg
  DepthWithPose.msg
  Pcd2WithPose.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs sensor_msgs geometry_msgs nav_msgs
)

file(GLOB python_scripts_DIR
    "scripts/*.py"
)

catkin_install_python(PROGRAMS ${python_scripts_DIR}
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

