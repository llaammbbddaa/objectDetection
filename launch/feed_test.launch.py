"""Run the folder detector and video feeder; keep the detector alive to drain backlog."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessIO
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {
        'watch_dir': 'camera_feed_images', 'output_dir': 'annotated_output',
        'tiling': 'auto', 'conf_threshold': '0.25', 'nms_threshold': '0.45',
        'loop': 'false', 'dump_all': 'false',
    }
    args = [DeclareLaunchArgument('video_path', description='Input video path')]
    args += [DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()]
    value = LaunchConfiguration
    detector = Node(
        package='yolo26_ros', executable='detector_node', output='screen',
        additional_env={'RCUTILS_LOGGING_USE_STDOUT': '1', 'RCUTILS_LOGGING_BUFFERED_STREAM': '0'},
        parameters=[{
            'watch_dir': value('watch_dir'), 'output_dir': value('output_dir'),
            'tiling': value('tiling'),
            'conf_threshold': ParameterValue(value('conf_threshold'), value_type=float),
            'nms_threshold': ParameterValue(value('nms_threshold'), value_type=float),
        }])
    feeder = Node(
        package='yolo26_ros', executable='video_feeder_node', output='screen',
        parameters=[{
            'video_path': value('video_path'), 'output_dir': value('watch_dir'),
            'loop': ParameterValue(value('loop'), value_type=bool),
            'dump_all': ParameterValue(value('dump_all'), value_type=bool),
        }])
    # Model warm-up may take several seconds. Start feeding only after inotify is ready.
    pending = bytearray()
    started = False

    def on_output(event):
        nonlocal started
        pending.extend(event.text)
        if not started and b'Ready: watching ' in pending:
            started = True
            return [LogInfo(msg='Detector ready; starting video feeder'), feeder]
        if len(pending) > 8192:
            del pending[:-4096]
        return None

    handler = RegisterEventHandler(OnProcessIO(target_action=detector, on_stdout=on_output, on_stderr=on_output))
    return LaunchDescription(args + [handler, detector])
