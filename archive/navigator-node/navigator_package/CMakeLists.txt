cmake_minimum_required(VERSION 3.0.2)
project(navigator_package)

## Compile as C++11, supported in ROS Noetic
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  nav_msgs
  octomap_msgs
  tf
  tf2
  tf2_ros
  tf2_geometry_msgs
)

find_package(octomap REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs nav_msgs octomap_msgs tf tf2 tf2_ros tf2_geometry_msgs
  DEPENDS octomap
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
)

add_executable(navigator_node src/navigator_node.cpp)

target_link_libraries(navigator_node
  ${catkin_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
)

add_dependencies(navigator_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
