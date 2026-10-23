# YOLO26 ROS 2 detector and simulated drone camera

This repository is the `yolo26_ros` ROS 2 package. It provides a reusable C++ detector, a folder-watching detector node, a video feeder node, and the original command-line detector. The test rig measures how well CPU inference keeps up with a simulated camera, including time spent waiting behind other frames.

The supported container environment is **Ubuntu 22.04 Jammy, ROS 2 Humble, C++17, and custom OpenCV 4.11**. The former Ubuntu 24.04 base was changed to 22.04 to match Humble. (notably, the other branches do not involve ros2, so the ubuntu 24 was just the version i happened to be running) Python is used for the ROS launch file; model export also uses Python during the Docker build, but inference and both nodes are C++.

## Data flow

```text
video file -> video_feeder_node -> camera_feed_images/img1.png, img2.png, ...
                                          |
                                 inotify completion events
                                          |
                                  thread-safe FIFO
                                          |
                                  detector_node worker
                                          |
                  +-----------------------+----------------------+
                  |                                              |
 annotated_output/annotated_images/                  annotated_output/csv/
        annotated_img1.png                                  img1.csv
```

The nodes communicate through files. They publish no image or detection topics and need no DDS networking to exchange frames. The watched directory is created when absent. Files present before the watcher is registered are ignored. The launch file waits for detector readiness, including model warm-up and watcher registration, before starting the feeder.

## Build and run with Docker

From the repository root:

```bash
docker build -t yolo26-ros .

# Put drone.mp4 in the current directory first.
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 launch yolo26_ros feed_test.launch.py video_path:=/data/drone.mp4
```

The first build compiles OpenCV and downloads/exports YOLO26n, so it takes time and requires internet access. Later builds reuse cached stages. Use `--build-arg END2END=False` if the default end-to-end export is incompatible with OpenCV. `OPSET` defaults to 12.

The working directory is `/data`. Relative paths refer to that directory. Mounting the current directory makes the feed and results appear on the host. `--user` gives generated files your UID/GID. `ROS_LOG_DIR=/tmp/roslog` allows ROS logging without a writable home directory.

When the feeder finishes, the detector remains alive to finish its backlog. Wait for output counts to match, then press Ctrl+C to get the detector summary. Pressing Ctrl+C immediately abandons waiting frames rather than draining the queue.

Run the two nodes independently in separate terminals, starting the detector first and waiting for `Ready: watching`:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 run yolo26_ros detector_node

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 run yolo26_ros video_feeder_node --ros-args -p video_path:=/data/drone.mp4
```

For a fast backlog stress run:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 launch yolo26_ros feed_test.launch.py video_path:=/data/drone.mp4 dump_all:=true
```

## Detector node parameters

Pass parameters after `--ros-args`, for example `-p tiling:=none`.

| Parameter | Default | Meaning |
|---|---|---|
| `watch_dir` | `camera_feed_images` | Directory watched for completed arrivals |
| `output_dir` | `annotated_output` | Parent of `annotated_images/` and `csv/` |
| `model_path` | `yolo26.onnx` | ONNX model path |
| `conf_threshold` | `0.25` | Finite confidence threshold in [0,1] |
| `nms_threshold` | `0.45` | Finite NMS IoU threshold in [0,1] |
| `tiling` | `auto` | `auto`, `none`, or positive `WIDTHxHEIGHT` |

Startup rejects invalid paths, invalid thresholds/tiling, an unloadable model, or an output directory equal to or below the watched directory. Paths are resolved using `std::filesystem::weakly_canonical`, including existing symlink components. Configure parameters before startup; changing them dynamically does not reconfigure the running worker.

`auto` calculates `ceil(image_width/640)` columns and `ceil(image_height/640)` rows for each image. Thus 1920×1080 uses 3×2 tiles. `none` runs one letterboxed inference. `1280x720` fixes the layout at 2×2 for all input sizes; it does not resize the source to that resolution. Tiles partition the actual image evenly. A 1×1 layout takes the whole-image path and skips merge NMS. Each distinct image size is logged once with its layout.

Only `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, and `.ppm` arrivals are accepted, case-insensitively. Dot-files and directories are ignored. The watcher uses `IN_CLOSE_WRITE` and `IN_MOVED_TO`, rather than events that fire during writing. Decode failures get an initial attempt plus up to three retries, separated by 50 ms. A bad frame is logged and counted as failed; the next queued frame still runs.

The watcher records a steady-clock arrival timestamp and queues the filename. A filename already waiting in the queue is not queued again. One worker performs all inference, in arrival order. The ROS executor never performs inference. There is no queue limit or deliberate frame-dropping policy during normal operation. Unique filenames are required to preserve every frame: repeated overwrites of a waiting filename are coalesced, and queued filenames refer to the current file contents rather than snapshots.

The kernel's inotify queue is finite. An overflow produces a warning that arrivals may have been lost. The application does not reconstruct lost arrivals, because scanning the directory cannot reliably distinguish them from startup files. Ensure adequate disk space and inotify capacity for demanding stress runs.

## Video feeder parameters

| Parameter | Default | Meaning |
|---|---|---|
| `video_path` | required | Readable input video |
| `output_dir` | `camera_feed_images` | Destination; created but never cleared |
| `frame_interval_ms` | `100` | Positive sampling and pacing interval, maximum one day |
| `file_prefix` | `img` | Nonempty filename prefix, without dot-prefix or path separators |
| `start_index` | `1` | Positive first filename index |
| `image_format` | `png` | `png` or `jpg` |
| `png_compression` | `1` | PNG encoding level 0–9 |
| `loop` | `false` | Reopen at EOF, keeping filename indices increasing |
| `dump_all` | `false` | Sample without real-time pacing |

The video is decoded sequentially. Sample times are 0, 100, 200 ms, etc. Each sample uses the newest decoded source frame whose timestamp is at or before the sample time. One look-ahead frame is retained so a future frame is never selected early. The first decoded frame is numbered zero, and timestamps are `frame_number * 1000 / fps` when FPS is finite and positive. Otherwise the feeder requires usable, strictly increasing `CAP_PROP_POS_MSEC` values beginning at zero. For FPS fallback, the last observed timestamp interval estimates the final frame's duration.

A source below 10 FPS at the default interval repeats source frames; known low FPS produces one startup warning. Sampling stops at the end of the last frame's presentation interval. A 10-second CFR video produces approximately 100 default samples. Looping reopens the source and restarts video sample times without resetting output indices.

A wall timer writes one sample per tick; the first tick occurs after one interval and contains the sample from video time zero. Slow ticks are warned about. Frames are not skipped to catch up, so a slow encoder stretches the simulated feed. `dump_all` uses a minimal-period timer to emit samples as quickly as decoding and disk writes allow, while still returning to the executor between samples.

Startup logs include path, resolution, FPS, known frame count and interval. Progress is logged every 50 output frames. Normal EOF prints `Finished: N frames written`, cancels the timer and shuts down the feeder. Fatal decode/write errors are logged and return a failing exit status.

**Restart behavior:** the feeder never clears its output folder. Default names start again at `img1.png`, so rerunning overwrites existing inputs. Each rename is a new arrival for a running detector, which overwrites corresponding results. Use a fresh directory or `-p start_index:=500` to avoid collisions. If starting a new detector, existing files are ignored until overwritten or new files arrive.

## Launch arguments

`feed_test.launch.py` accepts required `video_path` and optional `watch_dir`, `output_dir`, `tiling`, `conf_threshold`, `nms_threshold`, `loop`, and `dump_all`, with the defaults above. `watch_dir` is passed as the feeder's `output_dir`. Thresholds are explicitly typed as floats and booleans as booleans. Other parameters, such as `model_path`, `start_index`, and compression, can be configured through independent `ros2 run` commands.

The launch readiness handler accumulates detector process output until it sees `Ready: watching`. It then starts the feeder exactly once. This avoids losing frames while the detector loads or warms up. If detector startup fails, the feeder is not started. Stop the launch after reviewing that startup error.

## Outputs and atomic file handling

For input `img12.png`, the node writes:

```text
annotated_output/annotated_images/annotated_img12.png
annotated_output/csv/img12.csv
```

The annotated image keeps the input extension. Each file is encoded to a hidden temporary sibling with a valid extension, then renamed over its destination. The detector ignores the feeder's hidden temporary files; the final rename triggers `IN_MOVED_TO`. Temporary files are removed after exceptions. The CSV and image are individually atomic, not a two-file transaction: a failure between them can leave only one completed output. Atomic rename prevents readers from seeing a partially encoded file; it is not a power-loss durability guarantee.

Different input names with the same stem, such as `img1.png` and `img1.jpg`, share the CSV destination. Use unique stems. Avoid concurrent producers writing the same target names: their deterministic temporary filenames would conflict.

CSV columns remain:

```text
x1,y1,x2,y2,confidence,class_id,class_name,resolutionX,resolutionY,detection_time
```

Coordinates are in original-image pixels and are written with two decimal places. Confidence uses four decimal places. Class IDs refer to the 80 COCO classes; unknown IDs use `unknown`. Resolution is the original width/height. `detection_time` is the worker-measured `detect()` duration in milliseconds, written with three decimal places. Zero detections still produces the header and an annotated copy of the source image.

Annotations preserve the original style: green one-pixel rectangles, a green label background, and black anti-aliased text containing class name and confidence.

## Measuring throughput and latency

Each successfully processed frame has one INFO line:

```text
[img12.png] detections=3 detect=212.400ms queue_wait=1380.200ms total=213.100ms latency=1593.300ms queue_depth=7
```

| Field | Measurement |
|---|---|
| `detect` | Only the `Detector::detect()` call: all tile inference and merge NMS |
| `queue_wait` | Arrival event timestamp to worker dequeue |
| `total` | Dequeue to both outputs written, including decode and annotation |
| `latency` | Arrival to outputs complete; `queue_wait + total` |
| `queue_depth` | Frames waiting immediately after dequeue |

Arrival timestamps measure when the watcher receives an event, not source-video capture or the exact filesystem write instant. All durations use a monotonic steady clock. CSV detection time and logged detection time use the same measurement. Model warm-up is excluded.

A warning fires when waiting frames first exceed 10, then 20, 40, etc. Growing queue depth and latency indicate inference is falling behind. Shutdown prints successful/failed/unprocessed counts, mean/p95/max detection time and latency, and maximum queued frames. P95 uses the nearest-rank convention; statistics cover successful frames. The implementation retains duration samples for an exact shutdown percentile, so statistics memory grows with processed frames; queue memory grows with backlog. There is no claim of bounded memory for indefinitely long runs.

Ctrl+C stops accepting work, wakes `poll()` through `eventfd`, wakes the queue condition variable and joins both threads. Waiting frames are counted as unprocessed. An active inference/write finishes first. Idle shutdown is immediate; the approximately two-second target depends on that active OpenCV call completing, since OpenCV CPU inference cannot safely be interrupted mid-call.

## Reusable inference library

`include/yolo26_ros/yolo26_detector.hpp` exposes:

| API | Role |
|---|---|
| `Detection` | Box corners, confidence and class ID |
| `TileLayout` | Number of columns/rows; `{0,0}` requests a whole-image pass |
| `Resolution`, `parseResolution()` | Parse positive `WIDTHxHEIGHT` input |
| `autoLayout()`, `layoutFromResolution()` | Derive tile counts using 640-pixel cells |
| `Detector(modelPath)` | Load ONNX once, set OpenCV/CPU backend, warm up on a blank image |
| `Detector::detect()` | Infer a full image with optional tiling |
| `Detector::detectLegacy()` | Preserve the CLI's historical explicit-1×1 merge behavior |
| `annotate()` | Return an annotated clone without modifying input |
| `writeCsv()` | Write the existing CSV schema; return success/failure |

`cv::dnn::Net` is used by only one thread. Library functions throw on inference failures; node-level code handles each frame's errors and owns all measurement. Swapping the detector implementation therefore preserves timing boundaries.

Preprocessing letterboxes each tile to 640×640 with gray padding, scales pixel values by 1/255, and converts BGR to RGB. End-to-end `[1,300,6]` output supplies final corner boxes, confidence and class ID, so single-pass inference does not apply another NMS. Classic `[1,84,8400]` output is transposed, best class scores are selected, center/size boxes become corners, and NMS removes overlaps. Both paths undo letterbox scaling/padding and clamp to the source tile.

Tiled inference translates tile coordinates back to the full image and performs the original class-agnostic merge NMS. The port preserves integer box conversion, score filtering and decoder behavior rather than changing model semantics.

## Original CLI

The executable remains `yolo26_detect`, available on the container PATH and through `ros2 run yolo26_ros yolo26_detect`.

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  yolo26_detect photo.jpg 0.25 0.45

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  yolo26_detect photo.jpg --resolution 1920x1080

docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  yolo26_detect images csv_output annotated_output 0.25 0.45
```

Single-image CLI output is `annotated.jpg` and `detections.csv` in the working directory. Directory mode sorts paths and writes `<stem>.csv` and `<stem>_annotated<ext>`; its historical default is 2×2 tiling. Explicit `--resolution` derives tile counts from the given size. This differs intentionally from the ROS node's `auto` default and `annotated_<stem>` naming. The CLI prints detection counts and output paths. Unreadable images are reported and skipped in directory mode, with a failing exit status if any fail.

The entrypoint sources ROS and the installed workspace, links `/opt/model/yolo26.onnx` into the working directory if no model is present, and executes the command. A locally supplied `yolo26.onnx` takes precedence. Existing image-first Docker invocations such as `... yolo26-ros photo.jpg` remain supported for paths that exist. With no arguments the container lists package executables.

## Build system and Docker stages

`package.xml` is the root ROS package manifest. `CMakeLists.txt` uses `ament_cmake`, C++17, `rclcpp`, threads and the custom OpenCV components. `yolo26_core` is static and links core/imgproc/imgcodecs/dnn; only the feeder additionally links videoio. ROS executables install below `lib/yolo26_ros`; headers install under `include`, the archive under `lib`, and launch files under `share/yolo26_ros`.

Docker stages:

1. `opencv-build`: Ubuntu 22.04 compiles OpenCV 4.11, including videoio with FFmpeg and without GStreamer.
2. `app-build`: `ros:humble-ros-base` builds `/ws/src/yolo26_ros` with colcon and explicit `OpenCV_DIR=/opt/opencv/lib/cmake/opencv4`.
3. `model-export`: isolated Python 3.11, CPU PyTorch and Ultralytics export YOLO26n to ONNX. These tools are not copied into runtime. Export dependencies remain unpinned as in the original build.
4. `runtime-base`: ROS base plus custom OpenCV shared libraries and Jammy JPEG/PNG/FFmpeg dependencies. `LD_LIBRARY_PATH` selects `/opt/opencv/lib`.
5. `smoke-test`: runs the CLI on a generated blank PPM using the exported model; runs `ldd` for all three executables and rejects unresolved libraries or OpenCV outside `/opt/opencv/lib`.
6. `runtime`: copies the install tree, model, smoke-test success marker and entrypoint. Depending on the marker ensures the final build cannot skip the smoke test.

**OpenCV conflict warning:** do not install ROS desktop, `cv_bridge`, `vision_opencv`, or `image_transport` into this package environment. They can bring in Jammy's system OpenCV 4.5.4, whose importer cannot run this model. The build prints the chosen OpenCV version/location. Keep ROS base, custom OpenCV 4.11, and the explicit CMake path. Jammy uses `libpng16-16`, not Noble's `libpng16-16t64`.

## Native build

Use a Humble workspace on Ubuntu 22.04 with OpenCV 4.11 installed separately, including FFmpeg-backed videoio:

```bash
. /opt/ros/humble/setup.bash
# Repository should be at <workspace>/src/yolo26_ros.
cd <workspace>
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
. install/setup.bash
export LD_LIBRARY_PATH=/opt/opencv/lib:${LD_LIBRARY_PATH:-}
ros2 run yolo26_ros detector_node --ros-args -p model_path:=/absolute/path/yolo26.onnx
```

Export a model separately if not using Docker:

```bash
pip install ultralytics onnx onnxslim
yolo export model=yolo26n.pt format=onnx opset=12 simplify=True
mv yolo26n.onnx yolo26.onnx
```

Add `end2end=False` for the classic layout. The old plain CMake build now requires the ROS/ament environment; the executable's argument interface remains unchanged.

## Repository map

| File | Purpose |
|---|---|
| `src/yolo26_detector.cpp` | Original letterbox, decoding, tiling, merge NMS, CSV and drawing logic |
| `src/yolo26_cli.cpp` | CLI argument parsing, directory traversal and library calls |
| `src/file_watcher.cpp` | Linux inotify event parsing, poll and eventfd shutdown |
| `src/detector_node.cpp` | Parameter validation, queue/thread lifecycle, retry logic, atomic outputs and metrics |
| `src/video_feeder_node.cpp` | Sequential video sampling, wall timer, encoding and looping |
| `include/yolo26_ros/yolo26_detector.hpp` | Public inference data types and API |
| `include/yolo26_ros/file_watcher.hpp` | Watcher callback API and ownership |
| `include/yolo26_ros/fs_utils.hpp` | Canonical paths, containment check and atomic-write helper |
| `launch/feed_test.launch.py` | Combined launch and readiness sequencing |
| `CMakeLists.txt`, `package.xml` | Build, dependencies and ROS installation |
| `Dockerfile`, `entrypoint.sh` | Build/export/runtime environment and command dispatch |
| `tests/video_sampling_checks.cpp` | Exact-pixel comparison against expected source timestamps |
| `tests/VALIDATION.md` | Recorded build and runtime verification results and limits |
| `tests/fixtures/` | Retained generated AVI clips for repeatable tests |
| `tests/core_checks.cpp` | Model-independent core and Linux filesystem-event checks |
| `tests/integration.py` | Container integration checks and generated video fixture |
| `ROS2_INSTRUCTIONS.md` | Original requirements and acceptance test plan |
| `README.md` | Quick entry point and ROS overview |
| `readme.md` | This detailed package guide |
| `.gitignore` | Excludes model weights, build/install/log and generated feed/output folders |

The former `main.cpp` has been split into library and CLI sources. Existing tracked example outputs are not regenerated automatically by the build.

## Verification and troubleshooting

The build smoke test verifies that the exported model runs and all installed executables resolve the custom OpenCV. Runtime acceptance tests are separate:

```bash
# Run after building the image; generated fixtures/results stay under a temporary directory.
python3 tests/integration.py
```

Model-independent checks can also run on a Linux host with OpenCV development files:

```bash
g++ -std=c++17 -Iinclude $(pkg-config --cflags opencv4) \
  src/yolo26_detector.cpp src/file_watcher.cpp tests/core_checks.cpp \
  $(pkg-config --libs opencv4) -pthread -o /tmp/yolo26-core-checks
/tmp/yolo26-core-checks
```

See [tests/VALIDATION.md](tests/VALIDATION.md) for the completed verification record, including original-CLI parity and exact-pixel normal/low-FPS sampling checks.

For original CLI parity, compile the pre-refactor `main.cpp` from Git with the same OpenCV/model and compare CSVs on several sizes and tiling options, excluding only `detection_time`. See `ROS2_INSTRUCTIONS.md` for the complete manual acceptance plan, including real-time pacing, Ctrl+C during backlog and a 1080p clip.

If no images are processed, check the `Ready` log, watch path and extensions, and create a new arrival after startup. If inference fails, verify the model path and OpenCV version; rebuild with classic export if necessary. If CSV and image counts differ, inspect per-frame errors and remaining backlog before stopping. For high latency, compare queue depth and detect time: the rig intentionally queues work instead of silently dropping older frames. For a video-open failure, check that the path is inside the mounted directory and its codec is supported by FFmpeg.
