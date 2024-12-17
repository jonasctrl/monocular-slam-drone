FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV QT_X11_NO_MITSHM=1
ENV LANG C.UTF-8
ENV GDK_BACKEND=x11
ENV ROS_DISTRO=noetic

RUN apt-get update

RUN apt-get install -y curl gnupg2 lsb-release && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

RUN apt-get update

RUN apt-get install -y \
    git \
    build-essential \
    cmake

RUN apt-get install -y \
    libgl1-mesa-glx \
    libboost-all-dev

RUN apt-get install -y \
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
    ros-${ROS_DISTRO}-rviz

RUN apt-get install -y \
    python3-catkin-tools \
    python3-pip


RUN apt-get install -y \
    libpython3-dev \
    python3-rosdep

RUN python3 -m pip install --upgrade pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip3 install transformers==4.45.0
RUN pip3 install msgpack-rpc-python
RUN pip3 install numpy scipy numpy-quaternion
RUN pip3 install airsim
RUN pip3 install numba
RUN pip3 install pyfiglet
RUN pip3 install termcolor

RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make'

WORKDIR /catkin_ws/src
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash \
    && catkin_create_pkg drone-node rospy nav_msgs std_msgs geometry_msgs'

RUN rosdep init && \
    rosdep update

WORKDIR /catkin_ws
RUN rosdep install --from-paths src --ignore-src -r -y

RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && catkin_make"

RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
ENV ROS_PACKAGE_PATH=/catkin_ws/src

WORKDIR /catkin_ws/src
COPY ./drone-node/launch ./drone-node/launch
COPY ./drone-node/rviz ./drone-node/rviz
COPY ./drone-node/src ./drone-node/src
COPY ./drone-node/CMakeLists.txt ./drone-node/CMakeLists.txt
COPY ./drone-node/package.xml ./drone-node/package.xml

RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    cd /catkin_ws && catkin_make "

CMD ["/bin/bash", "-c", "source /catkin_ws/devel/setup.bash && roslaunch drone-node drone.launch"]
