# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set basic locale environment variables (ROS2 requires UTF-8)
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install basic tools needed to add the ROS repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc -o /tmp/ros.asc \
    && gpg --dearmor -o /etc/apt/trusted.gpg.d/ros.gpg /tmp/ros.asc \
    && rm /tmp/ros.asc \
    && echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list

# Declare the build argument for non-interactive installs
ARG DEBIAN_FRONTEND=noninteractive

# Install required packages (including ROS2 Humble)
RUN apt-get update && \
    DEBIAN_FRONTEND=$DEBIAN_FRONTEND apt-get install -y --no-install-recommends \
        locales \
        curl \
        gnupg2 \
        lsb-release \
        build-essential \
        cmake \
        git \
        wget \
        python3-pip \
        python3-dev \
        python3-colcon-common-extensions \
        python3-rosdep \
        ros-humble-ros-base \
        libgl1-mesa-glx \
        ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US en_US.UTF-8

# Install additional Python dependencies (including PyTorch with CUDA support)
RUN pip3 install --no-cache-dir \
    opencv-python==4.10.0.84 \
    tensorboardX==2.6.2.2 \
    numpy==2.2.2 \
    pandas==2.2.3 \
    tqdm==4.66.5 \
    matplotlib==3.10.0 \
    PyMCubes==0.1.6 \
    rich==13.8.1 \
    pysdf==0.1.9 \
    dearpygui==2.0.0 \
    packaging==23.1 \
    scipy==1.13.1 \
    lpips==0.1.4 \
    imageio==2.35.1 \
    PyYAML \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Create the ROS2 workspace directory and its src folder
ENV WORKSPACE=/nerf_ws
RUN mkdir -p ${WORKSPACE}/src

# Copy the ROS2 package from the build context
# Assumes the build context is set to the root of nerf_ws so that src/seg_package exists.
COPY src/seg_package ${WORKSPACE}/src/seg_package

# (Optional) Initialize rosdep – ignore errors if already initialized
RUN rosdep init || true && rosdep update || true

# Build the workspace using colcon (sourcing ROS2 environment in the same RUN command)
WORKDIR ${WORKSPACE}
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# Copy the entrypoint script (placed in nerf_ws/docker/entrypoint.sh)
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint to source ROS2 and workspace setup files before running commands
ENTRYPOINT ["/entrypoint.sh"]

# Default command: run the seg package node.
CMD ["ros2", "run", "seg_package", "seg_node"]
