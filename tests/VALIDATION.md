# Verification record

Verified on 2026-10-23 using the generated `yolo26-ros` image (Ubuntu 22.04, ROS 2 Humble, OpenCV 4.11, default YOLO26n end-to-end ONNX export).

- Full multi-stage Docker build passed, including the blank-PPM model smoke test.
- `ldd` checks passed for `yolo26_detect`, `detector_node`, and `video_feeder_node`: OpenCV resolved under `/opt/opencv/lib`, with no missing libraries.
- Model-independent C++ checks passed: tile layouts, malformed resolutions, containment, CSV header/zero detections, drawing without modifying input, existing-file ignore, ignored input types, close-write completion and atomic move arrivals. Host OpenCV was 4.10; runtime tests used the required 4.11.
- The pre-refactor CLI was compiled from the original source with the same OpenCV 4.11 as the new CLI. CSV rows matched in every column except `detection_time` for both original repository images, with no explicit tiling, `1280x720` (2×2), and `640x480` (explicit 1×1). Detection counts for the smaller image were 4/3/4; counts for the larger image were 11/16/11. This also checks preservation of the CLI's historical explicit-1×1 merge path.
- Container integration checks passed for startup-file ignore, close-write completion, move arrivals, ignored names/extensions/directories, and recovery after a corrupt PNG.
- The retained 21-second/30-FPS fixture generated exactly 210 samples. Stress mode processed every sample in index order with exactly one CSV and annotated image per input. Backlog warnings and timing fields were present.
- Repeating the feeder into the same directory caused overwrite and re-detection of all 210 frames.
- The one-second fixture produced 10 samples at approximately 100 ms intervals; observed modification-time intervals were 97.7–102.3 ms during this run. Timing varies with system load.
- Exact-pixel sample checks passed for both 30-FPS and 5-FPS one-second clips: every one of the 10 emitted PNGs matched the expected decoded source frame. The 5-FPS case repeated source frames and emitted the low-FPS warning.
- Feeder option checks passed for JPG encoding, custom `cam` prefix, `start_index=500`, and continuous indices across multiple video loops. Looping also shut down cleanly on Ctrl+C.
- The output-within-watch-directory configuration failed clearly at startup.
- A 1920×1080 input logged 3×2 for `auto`, 1×1 for `none`, and 2×2 for `1280x720`; outputs were produced in each mode.
- Combined launch waited for detector readiness, generated all 10 short-video outputs, and kept the detector alive after feeder EOF to allow backlog draining.
- Ctrl+C passed both idle and mid-backlog tests within the test's 2.5-second allowance. Mid-backlog shutdown reported remaining unprocessed frames; shutdown summaries included failures and timing statistics.

`tests/integration.py` regenerates the small fixtures using host FFmpeg and runs container checks with the host UID/GID. It uses temporary directories for runtime results and retains the AVI source fixtures under `tests/fixtures/`. `--extended-only` runs just the 1080p, combined-launch and backlog-shutdown cases.

`tests/video_sampling_checks.cpp` is an additional exact-pixel check: compile it against the custom OpenCV, run the feeder on a CFR source at its default 100-ms sampling interval, and pass the source video plus emitted-frame directory to the checker. It compares every sample against source frame `floor(sample_index * 0.1 * fps)`.

Limits: this record does not claim a separately built classic-export image, real drone footage performance, indefinite bounded memory, or a hard two-second deadline for an arbitrarily slow in-flight OpenCV call. The classic decoder was moved unchanged. Runtime statistics retain samples for exact p95, and filenames accumulate on disk by design. See `readme.md` for overflow, overwrite, and shutdown semantics.
