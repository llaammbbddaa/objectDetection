# YOLO26 object detection with ROS 2

A C++17 ROS 2 Humble package for OpenCV YOLO26 inference and a simulated drone camera feed. See [readme.md](readme.md) for the full package guide, parameters, implementation details, output formats and testing.

## ROS 2 test rig

```text
video -> video_feeder_node -> camera_feed_images/
                                      |
                               detector_node
                                      |
                 annotated_output/{annotated_images,csv}/
```

```bash
docker build -t yolo26-ros .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 launch yolo26_ros feed_test.launch.py video_path:=/data/drone.mp4
```

The feeder writes a sample every 100 ms. The detector processes completed new arrivals in FIFO order and logs detection time, queue wait, total processing time, arrival-to-output latency and queue depth. It summarizes mean/p95/max timing on shutdown. Let its backlog finish before Ctrl+C.

Existing inputs are ignored at detector startup. The feeder never clears its directory; restarting at index 1 overwrites old inputs and corresponding outputs. Use a fresh directory or `start_index` to preserve prior runs.

The original CLI remains available:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  yolo26_detect photo.jpg --resolution 1280x720
```

Single-image CLI output remains `annotated.jpg` and `detections.csv`. Directory mode accepts `images csv_output annotated_output [conf] [nms]` and defaults to 2×2 tiling; ROS defaults to per-image `auto` tiling. ROS annotated names use `annotated_<stem><ext>`.

## Project structure and code overview

The repository root is the `yolo26_ros` ament package. `src/yolo26_detector.cpp` contains the reusable inference library; `src/yolo26_cli.cpp` preserves CLI behavior. `src/file_watcher.cpp` implements inotify, `src/detector_node.cpp` manages queueing and measurement, and `src/video_feeder_node.cpp` samples video. Public headers are under `include/yolo26_ros/`, and `launch/feed_test.launch.py` starts both nodes after detector readiness.

Docker now uses **Ubuntu 22.04 rather than 24.04**, to match **ROS 2 Humble**. Keep custom **OpenCV 4.11** and **ros-base**; installing ROS desktop or cv_bridge can introduce incompatible system OpenCV 4.5.4. The Docker smoke test checks inference and shared-library resolution. Full parameter tables, independent-node Docker commands, log definitions, build stages and troubleshooting are in [readme.md](readme.md).
