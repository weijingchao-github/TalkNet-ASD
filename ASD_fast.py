import importlib.util
import os

import rospy
import rospkg
from cv_bridge import CvBridge


def _load_asdetector_module():
    package_path = rospkg.RosPack().get_path("active_speaker_detection")
    source_path = os.path.join(package_path, "scripts", "TalkNet", "ASDetector_1_0.py")
    module_name = "active_speaker_detection_asdetector_1_0_source"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "ASDetector"):
        raise ImportError(f"ASDetector not found in {source_path}")
    return module


_ASD_MODULE = _load_asdetector_module()
BaseASDetector = _ASD_MODULE.ASDetector
ActiveSpeakerAudio = _ASD_MODULE.ActiveSpeakerAudio
FaceBboxPerImageWithAudio = _ASD_MODULE.FaceBboxPerImageWithAudio
SimpleNamespace = _ASD_MODULE.SimpleNamespace
TalkNet = _ASD_MODULE.TalkNet
deque = _ASD_MODULE.deque
threading = _ASD_MODULE.threading


class ASDetectorFast(BaseASDetector):
    def __init__(self):
        # Model init
        self.active_speaker_detector = TalkNet()
        self.active_speaker_detector.loadParameters(
            os.path.join(os.path.dirname(__file__), "pretrain_TalkSet.model")
        )
        self.active_speaker_detector.eval()

        # others
        self.viz_flag = rospy.get_param("~viz_flag", False)
        self.count = 0
        self.algs_args = SimpleNamespace(
            crop_scale=0.40,
            min_ASD_greater_than_zero=rospy.get_param(
                "~min_ASD_greater_than_zero", 2
            ),
        )
        self.fast_window_seconds = rospy.get_param("~asd_fast_window_seconds", 0.5)
        self.audio_image_pub_frequency = rospy.get_param("/pub_frequency")
        self.per_seq_recv_times = max(
            1, int(round(self.audio_image_pub_frequency * self.fast_window_seconds))
        )
        self.recv_counter = 0
        self.do_ASD_flag = False
        self.seq_id = -1
        self.recv_msg_buffer = deque(maxlen=self.per_seq_recv_times)
        self.recv_msg_buffer_copy = None
        self.audio = None
        self.bridge = CvBridge()

        # ROS init
        rospy.Subscriber(
            "/face_and_person/face_detect_result_with_audio",
            FaceBboxPerImageWithAudio,
            self._process_recv_msg,
            queue_size=10,
        )
        self.pub_ASD_result = rospy.Publisher(
            "/ASD_fast/ASD_fast_result",
            ActiveSpeakerAudio,
            queue_size=10,
        )

        # loop
        self.thread_running = True
        self.loop_thread = threading.Thread(target=self._pipeline)
        self.loop_thread.start()


def main():
    rospy.init_node("active_speaker_detection_fast")
    ASD_detector = ASDetectorFast()
    try:
        rospy.spin()
    finally:
        ASD_detector.thread_shutdown()


if __name__ == "__main__":
    main()
