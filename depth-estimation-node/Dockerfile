# Use a CUDA-enabled base image for GPU support
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV ROS_DISTRO=noetic
ENV DEBIAN_FRONTEND=noninteractive

# Add the ROS repository to sources.list
RUN apt-get update && \
    apt-get install -y curl gnupg2 lsb-release && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Install necessary packages including ROS Noetic
RUN apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-pip \
    libgl1-mesa-glx \
    ros-${ROS_DISTRO}-catkin \
    ros-${ROS_DISTRO}-tf2-ros \
    ros-${ROS_DISTRO}-visualization-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-nav-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-camera-info-manager \
    ros-${ROS_DISTRO}-roslint \
    ros-${ROS_DISTRO}-rosbash \
    libglib2.0-0 \
    x11-apps \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pip and other dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install transformers==4.45.0

# Setup ROS workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws/src

RUN apt-get update && apt-get install -y \
    python3-rosdep

# Initialize rosdep
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
    rosdep init; \
    rosdep update; \
fi

# Copy the depth estimation package into the container
COPY ./depth_estimation_package /catkin_ws/src/depth_estimation_package

# Build the workspace
WORKDIR /catkin_ws
RUN rosdep install --from-paths src --ignore-src -r -y
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make"

# Setup environment for running ROS
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
ENV ROS_PACKAGE_PATH=/catkin_ws/src

# Copy launch files
COPY ./depth_estimation_package/launch /catkin_ws/src/depth_estimation_package/launch

# Default command
WORKDIR /catkin_ws/src/depth_estimation_package
CMD ["sleep", "infinity"]
