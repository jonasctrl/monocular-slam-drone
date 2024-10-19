# Base image with ROS Noetic Desktop Full
FROM osrf/ros:noetic-desktop-full

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ros-${ROS_DISTRO}-octomap \
    ros-${ROS_DISTRO}-octomap-msgs \
    ros-${ROS_DISTRO}-octomap-ros \
    ros-${ROS_DISTRO}-octomap-rviz-plugins \
    ros-${ROS_DISTRO}-tf \
    ros-${ROS_DISTRO}-tf2 \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-tf-conversions \
    && rm -rf /var/lib/apt/lists/*

# Create a catkin workspace and copy the package
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws/src

# Copy the C++ package source code into the workspace
COPY ./navigator_package /catkin_ws/src/navigator_package

# Build the workspace
WORKDIR /catkin_ws
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    catkin_make && \
    echo 'source /catkin_ws/devel/setup.bash' >> ~/.bashrc"

# Set the entrypoint to launch the navigation node
CMD ["/bin/bash", "-c", "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    source /catkin_ws/devel/setup.bash && \
    roslaunch navigator_package navigator.launch"]
