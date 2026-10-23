# Instructions: Port the YOLO26 Detector to ROS 2 and Add a Simulated Drone Camera Feed

These instructions describe how to turn the existing C++ object detection program (`main.cpp`) into a ROS 2 package containing two nodes:

1. **`detector_node`**: watches a folder for new images, runs YOLO26 detection on each one, and writes annotated images and CSV files to an output folder.
2. **`video_feeder_node`**: reads a video file and writes one frame every 100 ms into the watched folder, simulating a live drone camera.

The purpose of the whole setup is to **test the capabilities of the object detection system against a simulated live camera feed**. Every design choice below favors measurability (logging, ordering, no silently dropped frames) over cleverness.

---

## 1. Decisions already made

| Topic | Decision |
|---|---|
| ROS 2 distro / OS | **ROS 2 Humble Hawksbill on Ubuntu 22.04 (Jammy Jellyfish)** |
| Language | **Everything in C++** (nodes, library). The launch file is the one exception, because ROS 2 launch files are Python |
| "Subscribing" to a folder | The detector node does **not** use ROS topics. It watches the folder for new files (Linux `inotify`). No topics are published |
| Existing files at startup | **Ignored.** Only images that arrive after the detector starts are processed |
| Backlog policy | **Process every frame, in arrival order.** Never drop frames. Log queue depth and latency for every frame. (The detector may get faster later; these logs are how improvements will be measured.) |
| Tiling | A `tiling` parameter: `auto` (default), `none`, or `WIDTHxHEIGHT` |
| Video timing | Sample the video every 100 ms of video time and write files at a real-time 100 ms wall-clock pace |
| Output folder clearing | The feeder does **not** clear `camera_feed_images` on start |
| Folder locations | Parameters, resolved relative to the working directory |
| Runtime | Docker (extends the existing multi-stage `Dockerfile`) |

> **Version note.** Ubuntu 22.04's codename is *Jammy Jellyfish*, which is the Ubuntu paired with Humble. The existing `Dockerfile` uses Ubuntu 24.04, so its base images and one package name change (see section 7).

---

## 2. Target behavior at a glance

```
 drone.mp4
    |
    v
+---------------------+   writes img1.png, img2.png, ... every 100 ms
|  video_feeder_node  | ------------------------------------------+
+---------------------+                                           |
                                                                  v
                                                      camera_feed_images/
                                                                  |
                                                  inotify: new file appears
                                                                  v
                                                    +---------------------+
                                                    |    detector_node    |
                                                    +---------------------+
                                                                  |
                                           +----------------------+---------------------+
                                           v                                            v
                          annotated_output/annotated_images/              annotated_output/csv/
                              annotated_img1.png                              img1.csv
                              annotated_img2.png                              img2.csv
```

Folder structure produced:

```
camera_feed_images/
|_ img1.png
|_ img2.png
|_ ...

annotated_output/
|_ annotated_images/
|   |_ annotated_img1.png
|   |_ annotated_img2.png
|   |_ ...
|_ csv/
    |_ img1.csv
    |_ img2.csv
    |_ ...
```

Naming rules:

- Input frames: `<prefix><index><ext>`, e.g. `img1.png` (no zero padding, index starts at 1).
- Annotated image: `annotated_<stem><ext>`, e.g. `annotated_img1.png` (same extension as the input).
- CSV: `<stem>.csv`, e.g. `img1.csv`.

> The current `main.cpp` names annotated files `<stem>_annotated<ext>`. The ROS nodes must use the **prefix** style above (`annotated_<stem><ext>`). The standalone CLI may keep its old naming.

---

## 3. Package layout

Make the **repository root the ROS 2 package** (put `package.xml` at the root). Docker copies the repo into `/ws/src/yolo26_ros`. Package, project and `package.xml` name: `yolo26_ros`. (Names are suggestions; change them consistently if you prefer others.)

```
.
├── package.xml
├── CMakeLists.txt                  # rewritten for ament_cmake (section 6)
├── Dockerfile                      # extended (section 7)
├── README.md                       # add a ROS 2 section (section 10)
├── include/yolo26_ros/
│   ├── yolo26_detector.hpp         # detection library API
│   ├── file_watcher.hpp            # inotify wrapper
│   └── fs_utils.hpp                # path / atomic-write helpers
├── src/
│   ├── yolo26_detector.cpp         # logic moved out of main.cpp
│   ├── file_watcher.cpp
│   ├── detector_node.cpp           # ROS 2 node 1
│   ├── video_feeder_node.cpp       # ROS 2 node 2
│   └── yolo26_cli.cpp              # the original main(), now a thin wrapper
├── launch/
│   └── feed_test.launch.py         # starts both nodes
└── .gitignore                      # add camera_feed_images/, annotated_output/, build/, install/, log/
```

---

## 4. Step 1: Refactor `main.cpp` into a reusable library

The ROS node must not shell out to the CLI. The detection logic becomes a class that is **constructed once** (loading `yolo26.onnx` a single time) and then called per image.

### 4.1 What moves where

| Currently in `main.cpp` | Goes to | Notes |
|---|---|---|
| `INPUT_SIZE`, `CLASS_NAMES`, `letterbox()`, `Detection`, `makeTiles()`, `detectTile()` | `yolo26_detector.cpp` | Keep the decode logic **byte-for-byte equivalent** (end-to-end `[1,300,6]` layout and classic `[1,84,8400]` layout) |
| Tiling + cross-tile `cv::dnn::NMSBoxes` merge in `processImage()` | `yolo26_detector.cpp` | Keep as is, including class-agnostic NMS. Do not "improve" it during the port |
| CSV writing and annotation drawing in `processImage()` | `yolo26_detector.cpp` as separate functions | Keep the exact CSV header and drawing style (see 4.3) |
| `isDirectory`, `makeDirectory`, `joinPath`, `fileStem`, `fileExtension` | Replace with `std::filesystem` (C++17) in the nodes; the CLI may keep the old helpers | |
| `parseResolution()` | `yolo26_detector.cpp` (reused by the `tiling` parameter) | |
| `main()` | `yolo26_cli.cpp` | Behavior unchanged |

### 4.2 Suggested API

```cpp
namespace yolo26 {

struct Detection { float x1, y1, x2, y2; float confidence; int classId; };

struct TileLayout { int columns; int rows; };   // {0,0} means "no tiling"

class Detector {
public:
  explicit Detector(const std::string& modelPath);  // loads ONNX, sets OPENCV/CPU backend/target, runs a warm-up pass
  // Runs detection on one full image. Tiling is applied if layout.columns*layout.rows > 1.
  std::vector<Detection> detect(const cv::Mat& image, float conf, float nms,
                                TileLayout layout);
};

// Derives the layout from an image size using ceil(width/640) x ceil(height/640).
TileLayout autoLayout(const cv::Size& imageSize);
TileLayout layoutFromResolution(int width, int height);   // same rule, explicit size

cv::Mat annotate(const cv::Mat& image, const std::vector<Detection>& dets);
bool    writeCsv(const std::string& path, const std::vector<Detection>& dets,
                 const cv::Size& imageSize, double detectionTimeMs);

} // namespace yolo26
```

Requirements:

- **Warm-up.** The constructor runs one forward pass on a blank 640x640 image so the first real frame's latency is not inflated by lazy initialization.
- **Single-threaded use.** `cv::dnn::Net` is not safe to call from multiple threads. The node uses exactly one inference thread (section 5.1).
- **Swappable detector.** The detector may be replaced or improved later. Keep the node code depending only on `Detector::detect()`, `annotate()` and `writeCsv()`, and keep **all latency measurement in the node**, not inside the detector, so a faster detector drops in without touching the logging.
- Errors in `detect()` throw exceptions (`std::runtime_error` / `cv::Exception`) as today. The node catches them per image (section 5.1).

### 4.3 CSV format (unchanged)

```
x1,y1,x2,y2,confidence,class_id,class_name,resolutionX,resolutionY,detection_time
```

`detection_time` is the wall-clock time of the `detect()` call in milliseconds (same measurement as the original `detectionStart` to end, covering all tiles + merge NMS). A frame with zero detections still gets a CSV with only the header row.

### 4.4 Tiling rules

- `tiling=auto`: for each image, `columns = ceil(width/640)`, `rows = ceil(height/640)`, using that image's own dimensions.
- `tiling=none`: whole image, one letterboxed pass.
- `tiling=WIDTHxHEIGHT` (e.g. `1920x1080`): use `ceil(W/640) x ceil(H/640)` from the **given** size for every image.
- If the resulting layout is 1x1 (e.g. a 640x480 frame under `auto`), take the **non-tiled path**, so the extra merge-NMS pass does not alter end-to-end results.
- Do **not** use the CLI's old 2x2 directory default. `auto` replaces it.
- Log the layout once per distinct image size, e.g. `Image size 1920x1080 -> 3x2 tiles`.

### 4.5 Keep the CLI working

`yolo26_cli.cpp` builds the `yolo26_detect` executable with the same arguments and output as today. The README's existing usage must still work. It serves as the regression reference (section 9, test 1).

---

## 5. Step 2: The two nodes

### 5.1 `detector_node`

**Parameters**

| Parameter | Default | Meaning |
|---|---|---|
| `watch_dir` | `camera_feed_images` | Folder to watch for new images (created if missing) |
| `output_dir` | `annotated_output` | Parent output folder. `annotated_images/` and `csv/` are created inside it |
| `model_path` | `yolo26.onnx` | ONNX model, resolved relative to the working directory |
| `conf_threshold` | `0.25` | Must be in [0,1] |
| `nms_threshold` | `0.45` | Must be in [0,1]. Used for classic layout and tile merging |
| `tiling` | `auto` | `auto`, `none`, or `WIDTHxHEIGHT` |

Validate parameters at startup; exit with a clear error message on invalid values or an unloadable model.

**Threading model** (the executor thread must never block on inference):

```
 watcher thread  --(push filename + arrival time)-->  thread-safe FIFO queue  --> worker thread
 (inotify, poll)                                      (mutex + condition var)    (the ONLY thread
                                                                                  that uses the Detector)
```

- **Watcher thread**: blocks in `poll()` on the `inotify` fd plus a shutdown `eventfd`. Registers `IN_CLOSE_WRITE | IN_MOVED_TO` on `watch_dir`.
  - `IN_CLOSE_WRITE` fires when a writer finishes a file; `IN_MOVED_TO` fires when a file is renamed into the folder. Together they mean the file is **complete**. Do **not** use `IN_CREATE` or `IN_MODIFY`, which fire while a file is still being written.
  - Read events in a loop, handle `IN_Q_OVERFLOW` by logging a warning, and handle events whose name field spans multiple records correctly.
  - Ignore: directories, names starting with `.`, and anything whose lower-cased extension is not one of `.png .jpg .jpeg .bmp .tif .tiff .ppm`.
  - Record `steady_clock::now()` as the **arrival time** and push `(filename, arrival)` onto the queue. Skip a push if the same filename is already waiting in the queue.
- **Worker thread**: pops in FIFO order and for each image:
  1. `cv::imread`. If it fails or returns empty, retry up to 3 times with 50 ms between attempts (guards against a writer that is slightly late). Then log an error and skip the file.
  2. Compute the tile layout from `tiling` (section 4.4).
  3. `detect()`, then write outputs (below).
  4. Catch any exception, log it with the filename, and **continue with the next image**. One bad frame must never kill the node.
- **Shutdown**: register `rclcpp::on_shutdown` to set a stop flag, write to the `eventfd`, notify the queue's condition variable, and join both threads. Ctrl+C must exit cleanly within about 2 seconds, even mid-backlog (log how many items were left unprocessed).
- Do **not** process files already in `watch_dir` at startup.

**Outputs for each processed image `X<ext>`**

- `<output_dir>/annotated_images/annotated_X<ext>`
- `<output_dir>/csv/X.csv`

Write each output **atomically**: write to a hidden temp file in the same directory, then `std::filesystem::rename()` it over the final name. For images the temp name must keep a valid extension so `cv::imwrite` picks the right encoder, e.g. `.annotated_X.tmp<ext>`. For CSVs use `.X.csv.tmp`.

**Safety check**: if `output_dir` is inside `watch_dir` (resolve with `std::filesystem::weakly_canonical`), exit with an error. Otherwise the node would feed its own outputs back into itself.

**Per-frame logging** (required). One INFO line per processed image:

```
[img12.png] detections=3 detect=212.4ms queue_wait=1380.2ms total=1593.1ms latency=1593.1ms queue_depth=7
```

| Field | Definition |
|---|---|
| `detect` | Duration of the `detect()` call (same value written to the CSV) |
| `queue_wait` | Arrival time to the moment the worker dequeued it |
| `total` | Dequeue to outputs fully written (decode + detect + annotate + write) |
| `latency` | Arrival to outputs fully written (`queue_wait + total`). This is the number that shows whether the system keeps up with the feed |
| `queue_depth` | Images still waiting after this one was dequeued |

Also:

- Log a **WARN** when `queue_depth` is above a threshold that grows over time (e.g. first time it exceeds 10, then every doubling), saying the detector is falling behind the feed.
- On shutdown, log a **summary**: frames processed, frames failed, mean / p95 / max `detect` and `latency`, and max `queue_depth`. This summary is what allows before/after comparison when the detector is improved.

### 5.2 `video_feeder_node`

**Parameters**

| Parameter | Default | Meaning |
|---|---|---|
| `video_path` | *(required)* | Input video. Exit with a clear error if missing or unreadable |
| `output_dir` | `camera_feed_images` | Folder to write frames into (created if missing; **never cleared**) |
| `frame_interval_ms` | `100` | Sampling interval in video time and pacing interval in wall-clock time |
| `file_prefix` | `img` | Frames are named `<prefix><index><ext>` |
| `start_index` | `1` | First index used |
| `image_format` | `png` | `png` or `jpg` |
| `png_compression` | `1` | `IMWRITE_PNG_COMPRESSION` level 0-9. Low is used so encoding fits inside the 100 ms budget |
| `loop` | `false` | If true, restart the video at the end and **keep incrementing** the index (do not reset it) |
| `dump_all` | `false` | If true, write all sampled frames immediately without pacing (for stress tests) |

**Sampling rule (video time).** Sample times are `T_k = (k-1) * frame_interval_ms` for `k = 1, 2, ...` (first image is the frame at 0 ms, then 100 ms, 200 ms, ...). Frame `k` is the **most recent decoded frame whose timestamp is at or before `T_k`**, so a frame is never taken from the future.

- Decode **sequentially**, do not seek (seeking is slow and imprecise). Keep one look-ahead frame: read frames until one has a timestamp greater than `T_k`; the previous frame is the sample, and the look-ahead frame is carried into the next iteration.
- Frame timestamps: `frame_number / fps`, using `CAP_PROP_FPS`. If `fps` is zero or not finite, fall back to `CAP_PROP_POS_MSEC`. If neither works, exit with an error.
- If the source video fps is below `1000 / frame_interval_ms` (below 10 fps by default), the same source frame will be written more than once. Log one WARN about this at startup.

**Pacing (wall-clock).** Use an `rclcpp` wall timer with period `frame_interval_ms`; each tick produces exactly one image (unless `dump_all`). This simulates a live camera running at 10 images per second.

- If a tick takes longer than the interval (slow decode or write), log a WARN and carry on. Do not try to skip ahead.
- When the video ends and `loop` is false, log `Finished: N frames written`, cancel the timer, and call `rclcpp::shutdown()`.

**Atomic writes (important).** The detector must never see a half-written image. Write each frame to a hidden temp file in `output_dir` (keeping the extension, e.g. `.img12.tmp.png`), then `std::filesystem::rename()` it to `img12.png`. The rename triggers `IN_MOVED_TO` in the detector. The detector ignores dot-files, so the temp file is never picked up.

**Existing files.** Because the folder is not cleared, a frame whose name already exists will be **overwritten** (and the detector will treat it as a new arrival and overwrite its outputs). Document this in the README. Use `start_index` or a fresh folder between runs to avoid it.

**Logging.** INFO on start (video path, resolution, fps, frame count if known, interval), INFO every 50 frames written, INFO on finish. Include the video's resolution, since it determines the detector's `auto` tiling.

### 5.3 Launch file

`launch/feed_test.launch.py` starts both nodes. Launch arguments: `video_path` (required), `watch_dir`, `output_dir`, `tiling`, `conf_threshold`, `nms_threshold`, `loop`, `dump_all`, forwarded as node parameters. Both nodes should also work when started independently with `ros2 run`.

---

## 6. Step 3: Build system (`package.xml`, `CMakeLists.txt`)

- `package.xml` (format 3): `buildtool_depend` `ament_cmake`; `depend` `rclcpp`. **Do not** add `cv_bridge`, `image_transport`, `vision_opencv`, or any other dependency that pulls in the system OpenCV (see the warning below).
- `CMakeLists.txt`:
  - `CMAKE_CXX_STANDARD 17` (the original used 14; `rclcpp` on Humble and `std::filesystem` need 17).
  - `find_package(ament_cmake REQUIRED)`, `find_package(rclcpp REQUIRED)`, and `find_package(OpenCV 4.11 REQUIRED COMPONENTS core imgproc imgcodecs dnn videoio)`.
  - Print `message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")` so it is obvious which OpenCV was found.
  - Build a static library `yolo26_core` (detector + helpers), link `core imgproc imgcodecs dnn`. Only `video_feeder_node` links `videoio`.
  - Targets: `yolo26_cli` (installed as executable name `yolo26_detect`), `detector_node`, `video_feeder_node`. Link the nodes against `rclcpp`.
  - `install(TARGETS ...)` into `lib/${PROJECT_NAME}`, and `install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})`.
  - End with `ament_package()`.

> **Warning: OpenCV version conflict.** Ubuntu 22.04's system OpenCV is **4.5.4**, and the existing README says 4.5.4's ONNX importer **fails on YOLO26**. The project must keep using the custom **OpenCV 4.11** build from the Dockerfile. In the Docker build, make sure `-DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4` is passed to `colcon`, use `ros:humble-ros-base` (not `ros-humble-desktop`, which brings in the system OpenCV through `cv_bridge`), and verify at the end of the build that `ldd` of each executable shows `libopencv_*` resolving to `/opt/opencv/lib`, never `/usr/lib`.

---

## 7. Step 4: Update the Dockerfile

Keep the multi-stage structure and the smoke test. Required changes:

| Stage | Change |
|---|---|
| Global | `ARG UBUNTU_VERSION=22.04` (was 24.04). Add `ARG ROS_DISTRO=humble` |
| 1. `opencv-build` | Add FFmpeg dev packages: `libavcodec-dev libavformat-dev libswscale-dev libavutil-dev`. Change `-DBUILD_LIST=core,imgproc,imgcodecs,dnn` to `core,imgproc,imgcodecs,dnn,videoio` and add `-DWITH_FFMPEG=ON`. **Check the CMake summary shows `FFMPEG: YES`**, otherwise the video node will fail at runtime with a "could not open video" error |
| 2. `app-build` | Base on `ros:${ROS_DISTRO}-ros-base`. Install `build-essential cmake python3-colcon-common-extensions libjpeg-dev libpng-dev`. Copy `/opt/opencv` from stage 1. Copy the repo to `/ws/src/yolo26_ros`, then `colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4` |
| 3. `model-export` | **Unchanged** (separate `python:3.11-slim` stage; still supports `--build-arg END2END=False`) |
| 4. `runtime-base` | Base on `ros:${ROS_DISTRO}-ros-base`. Runtime packages for Ubuntu 22.04: `libjpeg-turbo8 libpng16-16 libgomp1 libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3`. **Note `libpng16-16` here, not `libpng16-16t64`** (the `t64` name only exists on 24.04). Copy `/opt/opencv/lib`, set `LD_LIBRARY_PATH=/opt/opencv/lib` |
| 5. `smoke-test` | Keep. Run the `yolo26_detect` CLI on the blank PPM exactly as today. Optionally also run `ldd` on `video_feeder_node` and fail if any `libopencv*` resolves outside `/opt/opencv/lib` or any library shows `not found` |
| 6. `runtime` | Copy `/ws/install` from stage 2 and the model from stage 3. Entry point script (below) |

**Entry point.** Replace `entrypoint.sh` with one that sources ROS, links the bundled model, then runs the requested command:

```sh
#!/bin/sh
. /opt/ros/humble/setup.sh
. /ws/install/setup.sh
[ -e yolo26.onnx ] || ln -s /opt/model/yolo26.onnx yolo26.onnx
exec "$@"
```

- `ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]` and a default `CMD` that prints usage or runs `ros2 launch yolo26_ros feed_test.launch.py`.
- Add `ENV ROS_LOG_DIR=/tmp/roslog`. With `--user "$(id -u):$(id -g)"` there is no writable home directory and ROS 2 would otherwise fail to create its log folder.
- `WORKDIR /data` stays, so `camera_feed_images` and `annotated_output` appear in the user's mounted folder.
- Keep the original CLI available: `docker run ... yolo26-ros yolo26_detect image.jpg`.

**Example runs** (image name `yolo26-ros`):

```bash
docker build -t yolo26-ros .

# Both nodes together (video must be in the mounted folder)
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 launch yolo26_ros feed_test.launch.py video_path:=/data/drone.mp4

# Or in two terminals sharing the same mounted folder:
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 run yolo26_ros detector_node
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/data yolo26-ros \
  ros2 run yolo26_ros video_feeder_node --ros-args -p video_path:=/data/drone.mp4
```

The two containers communicate only through the shared folder, so no DDS or network setup is needed.

---

## 8. Implementation order

1. Move the detection logic into `yolo26_detector.{hpp,cpp}`, thin `yolo26_cli.cpp`, switch to ament/C++17. Confirm the CLI output is identical to before (test 1).
2. Implement `file_watcher` and `detector_node`. Test by manually copying images into the folder (tests 2 to 6).
3. Update the Dockerfile for Humble, FFmpeg and `videoio`. Confirm `docker build` passes the smoke test.
4. Implement `video_feeder_node`. Test it alone (test 7).
5. Add the launch file and run the full pipeline (tests 8 to 10).
6. Update the README and `.gitignore`.

---

## 9. Test plan

Run these inside the container. For the video tests, use a short clip (about 10 s) with a known frame count, and keep the clip around as a fixture.

1. **Parity with the original.** Using the original binary and the new `yolo26_detect` CLI, run both on several test images (including one larger than 640 px, with and without `--resolution`). The CSVs must match in every column except `detection_time`.
2. **Naming and placement.** Copy `img1.png` into an empty `camera_feed_images` while the detector runs. Expect `annotated_output/annotated_images/annotated_img1.png` and `annotated_output/csv/img1.csv`, plus one log line in the format of section 5.1.
3. **Only new arrivals.** Put 3 images in the folder, *then* start the detector. Expect none to be processed. Add a 4th: only that one is processed.
4. **Incomplete files are not processed.** Write a large image slowly (e.g. chunked `dd` with sleeps, or a script that writes in pieces). The detector must act only after the file is closed. Also test a plain `mv` into the folder.
5. **Ignored inputs.** Drop in a `.txt` file, a dot-file, and a subdirectory. None may produce output or errors.
6. **Bad input.** Drop in a corrupt `.png`. Expect an error log line after the retries, no crash, and the next good image still processed.
7. **Feeder timing and content.** Run only the feeder on the 10 s clip. Expect about 100 files (`img1.png` ... `img100.png`), consecutive indices, and an interval between file modification times of about 100 ms (allow roughly +/-20 ms jitter). Open a few frames and confirm they come from the expected points in the video. The node should exit by itself after the last frame.
8. **End-to-end under load.** Run both nodes on the clip. Every input frame must have exactly one annotated image and one CSV (compare counts and names), with processing in index order. If the detector is slower than 100 ms per frame, the log must show `queue_depth` growing and the backlog WARN, and the shutdown summary must be printed.
9. **Stress.** Run the feeder with `dump_all:=true` on a longer clip (200+ frames). Confirm no frames are dropped, order is preserved, and memory use stays stable.
10. **Tiling modes.** Feed a 1920x1080 clip: `auto` should log a 3x2 layout. `none` should run one pass. `tiling:=1280x720` should log a 2x2 layout. Confirm output coordinates are in original-image pixels in every mode.
11. **Clean shutdown.** Press Ctrl+C (a) while idle and (b) mid-backlog. Both nodes exit within about 2 s without hanging threads. The detector prints its summary.
12. **Safety check.** Start the detector with `output_dir` inside `watch_dir`. It must refuse to start with a clear error.
13. **No-clear behavior.** Run the feeder twice into the same folder with default `start_index`. Confirm frames are overwritten and re-detected (documented behavior), and that `start_index:=500` avoids it.

---

## 10. Documentation to update

- **README.md**: add a "ROS 2 test rig" section covering the data flow diagram, parameters of both nodes, Docker run examples, the log format with the latency field definitions, the overwrite-on-restart behavior, and the OpenCV 4.11 / `ros-base` warning. Update "Project structure" and "Code overview". Note that the Ubuntu base moved from 24.04 to 22.04 and why.
- **.gitignore**: add `camera_feed_images/`, `annotated_output/`, `build/`, `install/`, `log/`.

---

## 11. Acceptance criteria

- [ ] `docker build -t yolo26-ros .` succeeds on Ubuntu 22.04 / ROS 2 Humble and the smoke test passes.
- [ ] The original CLI still works and its CSV output matches the pre-refactor output (except `detection_time`).
- [ ] `detector_node` loads the model once, watches `camera_feed_images`, and processes only images that arrive after startup, in arrival order, without ever processing a partially written file.
- [ ] Outputs match the folder structure and naming in section 2, and are written atomically.
- [ ] Per-frame logs include `detect`, `queue_wait`, `total`, `latency` and `queue_depth`, and a summary is printed on shutdown.
- [ ] `video_feeder_node` writes one frame per 100 ms of video time at a real-time wall-clock pace, with atomic writes and the naming from section 2.
- [ ] No frames are dropped under backlog, and a bad frame never crashes the detector.
- [ ] Both nodes shut down cleanly on Ctrl+C.
- [ ] `ldd` confirms every executable uses OpenCV 4.11 from `/opt/opencv/lib`.
- [ ] README and `.gitignore` are updated.

---

## 12. Out of scope (for now)

- Publishing detections or images on ROS topics (the detector only watches a folder).
- Processing images that already exist when the detector starts.
- A "latest frame only" mode that drops stale frames.
- GPU inference.
- Clearing `camera_feed_images` automatically.
