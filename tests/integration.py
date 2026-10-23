#!/usr/bin/env python3
"""Run with Python on the host; requires Docker and ffmpeg. Leaves fixture in tests/fixtures."""
import csv
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time


def wait_for(predicate, timeout=90):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError('Timed out waiting for expected result')


def inside():
    root = Path('/testdata')
    if '--extended-only' in sys.argv:
        extended(root)
        return
    feed = root / 'feed'
    feed.mkdir()
    ppm = b'P6\n96 64\n255\n' + bytes([40, 80, 120]) * (96 * 64)
    (feed / 'existing.ppm').write_bytes(ppm)
    log_path = root / 'detector.log'
    command = ['/ws/install/yolo26_ros/lib/yolo26_ros/detector_node', '--ros-args',
               '-p', f'watch_dir:={feed}', '-p', f'output_dir:={root / "output"}',
               '-p', 'model_path:=/opt/model/yolo26.onnx']
    with log_path.open('w') as log:
        detector = subprocess.Popen(command, stdout=log, stderr=log,
                                    env={**os.environ, 'RCUTILS_LOGGING_BUFFERED_STREAM': '0'})
        try:
            wait_for(lambda: 'Ready: watching' in log_path.read_text())
            outputs = root / 'output' / 'csv'
            assert not list(outputs.iterdir()), 'startup file was processed'
            (feed / 'ignored.txt').write_text('ignored')
            (feed / '.ignored.ppm').write_bytes(ppm)
            (feed / 'directory.ppm').mkdir()
            with (feed / 'slow.ppm').open('wb') as f:
                f.write(ppm[:100]); f.flush()
                time.sleep(0.3)
                assert not (outputs / 'slow.csv').exists()
                f.write(ppm[100:])
            wait_for(lambda: (outputs / 'slow.csv').exists())
            (feed / '.move.tmp.ppm').write_bytes(ppm)
            (feed / '.move.tmp.ppm').rename(feed / 'moved.ppm')
            (feed / 'bad.png').write_bytes(b'not an image')
            (feed / 'good.ppm').write_bytes(ppm)
            wait_for(lambda: (outputs / 'good.csv').exists())
            assert (root / 'output/annotated_images/annotated_moved.ppm').exists()
            assert not (outputs / 'bad.csv').exists()
            assert not (outputs / 'existing.csv').exists()
            print('PASS startup-ignore, completion events, rename, ignored inputs, corrupt-frame recovery', flush=True)

            # 21 seconds / 30 fps, sampled at 100 ms -> 210 outputs in one stress burst.
            feeder = ['/ws/install/yolo26_ros/lib/yolo26_ros/video_feeder_node', '--ros-args',
                      '-p', 'video_path:=/fixtures/clip.avi', '-p', f'output_dir:={feed}',
                      '-p', 'dump_all:=true']
            result = subprocess.run(feeder, capture_output=True, text=True, timeout=90)
            assert result.returncode == 0, result.stderr
            assert 'Finished: 210 frames written' in result.stdout + result.stderr
            wait_for(lambda: len(list(outputs.glob('img*.csv'))) == 210, timeout=180)
            expected = {f'img{i}' for i in range(1, 211)}
            assert {p.stem for p in outputs.glob('img*.csv')} == expected
            assert {p.stem.removeprefix('annotated_') for p in (root / 'output/annotated_images').glob('annotated_img*.png')} == expected
            text = log_path.read_text()
            order = [int(x) for x in re.findall(r'\[img(\d+)\.png\] detections=', text)]
            assert order == list(range(1, 211)), order
            assert 'falling behind' in text
            assert 'queue_wait=' in text and 'latency=' in text
            print('PASS 210-frame stress, complete outputs, FIFO, backlog warning and metrics', flush=True)

            # A second feeder run replaces existing inputs and causes new detections.
            result = subprocess.run(feeder, capture_output=True, text=True, timeout=90)
            assert result.returncode == 0
            wait_for(lambda: log_path.read_text().count('[img210.png] detections=') == 2, timeout=180)
            print('PASS overwrite-on-restart and re-detection', flush=True)
        finally:
            before = time.monotonic()
            detector.send_signal(signal.SIGINT)
            detector.wait(timeout=10)
            assert detector.returncode == 0
            assert time.monotonic() - before < 2.5
    text = log_path.read_text()
    assert 'Summary:' in text and 'failed=1' in text
    print('PASS clean shutdown and summary', flush=True)

    invalid = subprocess.run(command[:-4] + ['-p', f'output_dir:={feed / "nested"}'],
                             capture_output=True, text=True, timeout=15)
    assert invalid.returncode != 0
    assert 'output_dir must not be inside watch_dir' in invalid.stdout + invalid.stderr
    print('PASS output containment safety check', flush=True)

    # Ten real-time samples from a 1-second source.
    paced = root / 'paced'
    args = ['/ws/install/yolo26_ros/lib/yolo26_ros/video_feeder_node', '--ros-args',
            '-p', 'video_path:=/fixtures/short.avi', '-p', f'output_dir:={paced}']
    result = subprocess.run(args, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    frames = sorted(paced.glob('*.png'), key=lambda p: int(p.stem[3:]))
    assert len(frames) == 10
    intervals = [(b.stat().st_mtime_ns-a.stat().st_mtime_ns)/1e6 for a,b in zip(frames,frames[1:])]
    assert all(70 < v < 140 for v in intervals), intervals
    print(f'PASS pacing and sample count: intervals {min(intervals):.1f}..{max(intervals):.1f} ms', flush=True)


    extended(root)


def extended(root):
    ppm = b'P6\n96 64\n255\n' + bytes([40, 80, 120]) * (96 * 64)
    for tiling, expected in [('auto', '3x2'), ('none', '1x1'), ('1280x720', '2x2')]:
        directory = root / ('tiles-' + tiling)
        directory.mkdir()
        watch = directory / 'feed'
        log_file = directory / 'node.log'
        args = ['/ws/install/yolo26_ros/lib/yolo26_ros/detector_node', '--ros-args',
                '-p', f'watch_dir:={watch}', '-p', f'output_dir:={directory / "output"}',
                '-p', 'model_path:=/opt/model/yolo26.onnx', '-p', f'tiling:={tiling}']
        with log_file.open('w') as log:
            process = subprocess.Popen(args, stdout=log, stderr=log)
            try:
                wait_for(lambda: 'Ready: watching' in log_file.read_text())
                (watch / 'large.ppm').write_bytes(b'P6\n1920 1080\n255\n' + bytes([80, 80, 80]) * (1920*1080))
                wait_for(lambda: (directory / 'output/csv/large.csv').exists())
                assert f'1920x1080 -> {expected} tiles' in log_file.read_text()
            finally:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=10)
        assert process.returncode == 0
    print('PASS 1080p auto, none and explicit-resolution layouts', flush=True)

    launch_log = root / 'launch.log'
    with launch_log.open('w') as log:
        launch = subprocess.Popen(['ros2', 'launch', 'yolo26_ros', 'feed_test.launch.py',
                                   'video_path:=/fixtures/short.avi', f'watch_dir:={root / "launch-feed"}',
                                   f'output_dir:={root / "launch-output"}', 'dump_all:=true'],
                                  stdout=log, stderr=log, start_new_session=True)
        try:
            wait_for(lambda: len(list((root / 'launch-output/csv').glob('*.csv'))) == 10)
            assert launch.poll() is None, 'launch killed detector before backlog could drain'
        finally:
            os.killpg(launch.pid, signal.SIGINT)
            launch.wait(timeout=15)
    assert 'Detector ready; starting video feeder' in launch_log.read_text()
    assert 'Summary:' in launch_log.read_text()
    print('PASS combined launch readiness and backlog draining', flush=True)

    backlog_feed = root / 'backlog-feed'
    backlog_log = root / 'backlog.log'
    with backlog_log.open('w') as log:
        process = subprocess.Popen(['/ws/install/yolo26_ros/lib/yolo26_ros/detector_node', '--ros-args',
                                    '-p', f'watch_dir:={backlog_feed}', '-p', f'output_dir:={root / "backlog-output"}',
                                    '-p', 'model_path:=/opt/model/yolo26.onnx'], stdout=log, stderr=log)
        try:
            wait_for(lambda: 'Ready: watching' in backlog_log.read_text())
            for i in range(100):
                (backlog_feed / f'frame{i}.ppm').write_bytes(ppm)
            wait_for(lambda: 'falling behind' in backlog_log.read_text())
        finally:
            before = time.monotonic()
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
            assert time.monotonic() - before < 2.5
    assert re.search(r'unprocessed=[1-9]', backlog_log.read_text())
    print('PASS shutdown during backlog reports unprocessed frames', flush=True)


def host():
    repo = Path(__file__).resolve().parents[1]
    fixture = repo / 'tests/fixtures'
    fixture.mkdir(exist_ok=True)
    for name, duration in [('clip', 21), ('short', 1)]:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                        f'testsrc=size=96x64:rate=30:duration={duration}',
                        '-c:v', 'mjpeg', '-threads', '1', str(fixture / f'{name}.avi')], check=True)
    with tempfile.TemporaryDirectory(prefix='yolo26-integration-') as temp:
        subprocess.run(['docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
                        '-v', f'{repo}:/repo:ro', '-v', f'{fixture}:/fixtures:ro',
                        '-v', f'{temp}:/testdata', '-w', '/testdata', 'yolo26-ros',
                        'bash', '-c', 'python3 /repo/tests/integration.py --inside' + (' --extended-only' if '--extended-only' in sys.argv else '')], check=True)


if __name__ == '__main__':
    inside() if '--inside' in sys.argv else host()
