#!/bin/sh
set -e
. /opt/ros/humble/setup.sh
. /ws/install/setup.sh
[ -e yolo26.onnx ] || ln -s /opt/model/yolo26.onnx yolo26.onnx
# Preserve the original image-first Docker CLI usage as well as explicit commands.
case "${1:-}" in
  ros2|yolo26_detect|sh|bash|/bin/*|/usr/bin/*) ;;
  -h|--help|'') set -- ros2 pkg executables yolo26_ros ;;
  *) if [ -e "$1" ]; then set -- yolo26_detect "$@"; fi ;;
esac
exec "$@"
