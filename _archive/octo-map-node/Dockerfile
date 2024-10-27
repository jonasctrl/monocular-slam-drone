FROM ros:noetic

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:0 \
    QT_X11_NO_MITSHM=1 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    iputils-ping \
    liboctomap-dev \
    net-tools \
    python3-pip \
    ros-${ROS_DISTRO}-octomap \
    ros-${ROS_DISTRO}-octomap-msgs \
    ros-${ROS_DISTRO}-octomap-server \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-pcl-ros \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-octomap-rviz-plugins \
    ros-${ROS_DISTRO}-rviz \
    && pip3 install --no-cache-dir numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/catkin_ws/src
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash \
    && catkin_create_pkg octomap_launch roscpp octomap_ros octomap_server' \
    && mkdir -p octomap_launch/launch \
    && git clone https://github.com/OctoMap/octomap_ros.git

COPY launch octomap_launch/launch/

WORKDIR /opt/catkin_ws
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make'

RUN echo '#!/bin/bash\n\
    source /opt/ros/${ROS_DISTRO}/setup.bash\n\
    source /opt/catkin_ws/devel/setup.bash\n\
    exec "$@"' > /ros_entrypoint.sh \
    && chmod +x /ros_entrypoint.sh

COPY rviz octomap_launch/rviz/
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["roslaunch", "octomap_launch", "octomap_mapping.launch"]
