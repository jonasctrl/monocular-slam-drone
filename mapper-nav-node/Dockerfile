FROM ros:noetic-ros-core

ENV DEBIAN_FRONTEND=noninteractive
#ENV DISPLAY=:0
ENV QT_X11_NO_MITSHM=1
ENV LANG C.UTF-8
ENV GDK_BACKEND=x11

# Install essential dependencies
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
RUN pip3 install numpy scipy numpy-quaternion pandas Pillow
RUN apt-get update && \
    apt-get install -y \
    ros-${ROS_DISTRO}-rviz\
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make'

WORKDIR /catkin_ws/src
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash \
     && catkin_create_pkg nav rospy nav_msgs std_msgs geometry_msgs'


#RUN mkdir -p /catkin_ws/src/nav/scripts /catkin_ws/src/nav/launch /catkin_ws/src/nav/rviz
#WORKDIR /catkin_ws
#RUN catkin_make

#WORKDIR /catkin_ws/src

RUN rosdep init && \
    rosdep update

WORKDIR /catkin_ws
RUN rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make"

RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
ENV ROS_PACKAGE_PATH=/catkin_ws/src

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

#CMD ["sleep", "infinity"]
# Build the workspace
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /catkin_ws && catkin_make"

WORKDIR /catkin_ws/src
CMD ["/bin/bash", "-c", "source /catkin_ws/devel/setup.bash && roslaunch nav mapper_nav.launch"]
