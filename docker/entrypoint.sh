#!/bin/bash
set -e

# Source ROS and workspace
source "/opt/ros/humble/setup.bash"
source "${WORKSPACE}/install/setup.bash"

exec "$@"