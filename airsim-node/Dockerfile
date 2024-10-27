FROM ros:noetic-ros-core

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_X11_NO_MITSHM=1
ENV LANG C.UTF-8
ENV GDK_BACKEND=x11

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
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
    ros-${ROS_DISTRO}-rviz \
    python3-catkin-tools \
    libgl1-mesa-glx \
    git \
    build-essential \
    cmake \
    libboost-all-dev \
    libpython3-dev \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install msgpack-rpc-python
RUN pip3 install airsim

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws/src

RUN rosdep init && \
    rosdep update

COPY ./airsim_package /catkin_ws/src/airsim_package

WORKDIR /catkin_ws
RUN rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make"

RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
ENV ROS_PACKAGE_PATH=/catkin_ws/src

COPY ./airsim_package/launch /catkin_ws/src/airsim_package/launch

WORKDIR /catkin_ws/src/airsim_package

# CMD ["/bin/bash", "-c", "source /catkin_ws/devel/setup.bash && roslaunch airsim_package airsim_camera_xyz.launch"]
CMD ["sleep", "infinity"]
