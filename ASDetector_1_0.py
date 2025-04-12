"""写代码时需要考虑的问题
1. 假设在一个推理周期内，face的数量很少怎么办
    - 由self.algs_args.min_ASD_greater_than_zero参数解决这个问题，如果ASD到的每个推理周期内score>0的
      每个人的人脸数量小于这个参数，那就不认为这个推理周期内有人在说话
2. 如果没有检测到人说话，最后一张没有人脸，人脸图片张数少，检测到的说话的人脸图片张数少，则pub给ASR节点的
结果tracker_id设为-1
"""

import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)
sys.path.insert(
    0,
    "/home/zxr/Documents/wjc/HRI/project/gaze_point_select_ws/devel/lib/python3/dist-packages",
)

import copy
import threading
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np
import python_speech_features
import rospy
import scipy
import torch
from cv_bridge import CvBridge
from talkNet import talkNet as TalkNet

from active_speaker_detection.msg import ActiveSpeakerAudio
from face_and_person.msg import FaceBboxPerImageWithAudio


class ASDetector:
    def __init__(self):
        # Model init
        self.active_speaker_detector = TalkNet()
        self.active_speaker_detector.loadParameters(
            os.path.join(os.path.dirname(__file__), "pretrain_TalkSet.model")
        )
        self.active_speaker_detector.eval()
        # others
        self.viz_flag = False
        self.count = 0  # for viz
        self.algs_args = SimpleNamespace(
            crop_scale=0.40,
            # smooth_window_size=5,
            min_ASD_greater_than_zero=5,
        )
        self.audio_image_pub_frequency = rospy.get_param("/pub_frequency")
        duration = rospy.get_param("/llm_inferecnce_duration")
        self.per_seq_recv_times = int(self.audio_image_pub_frequency * duration)
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
            "ASD_result",
            ActiveSpeakerAudio,
            queue_size=10,
        )
        # loop
        self.thread_running = True
        self.loop_thread = threading.Thread(target=self._pipeline)
        self.loop_thread.start()

    def thread_shutdown(self):
        self.thread_running = False
        self.loop_thread.join()

    def _pipeline(self):
        while self.thread_running:
            if not self.do_ASD_flag:
                time.sleep(0.001)
                continue
            self.do_ASD_flag = False
            self.audio = None

            # 如果最新的一帧里没有人脸，那就不进行ASD
            # 只对最后一帧画面有的人脸进行ASD
            # 这样有个好处，在这1.5s如果有人边说话边走出摄像头视野范围，那么ASD匹配不到结果
            # 就不会识别这段语音来目光注视这个人
            # 坏处就是ASR会少记录1.5s的信息
            # 连续检的好处就是能够把这1.5s ASR的内容记录，而且能照顾到最后一帧视野中没有人的情况
            # 人走着走着进柱子后面被挡住了，转身看不见脸了属于corner case，先尝试着解决general case
            # 后面看看要不要做运动预测1.5s后人的位置
            current_msg = self.recv_msg_buffer_copy[-1]
            if len(current_msg.bboxes_xyxy_and_ids) == 0:
                self._pub_terminate_ASR_signal()
                continue

            # track faces
            track_results = self._get_face_track_result()

            # crop faces and audios
            cropped_face_images_and_audios, visualization_bboxes = (
                self._crop_face_and_audio(track_results)
            )

            # detect active speaker
            scores_multiple_people = self._detect_active_speaker(
                cropped_face_images_and_audios
            )

            # pub ASD result
            max_times_track_id = self._pub_ASD_result(scores_multiple_people)

            if self.viz_flag:
                self._visualization(max_times_track_id, track_results)

            # # visualization
            # self._viz_draw_bbox_and_text_on_image(
            #     scores_multiple_people,
            #     visualization_bboxes,
            #     current_image["color_image"],
            # )
            # cv2.imshow("image", current_image["color_image"])
            # cv2.waitKey(1)

            # # Compute and pub active speaker's position in Camera 3D Coordinates
            # self._compute_and_pub_active_speaker_position(
            #     scores_multiple_people,
            #     visualization_bboxes,
            #     current_image["depth_image"],
            # )

    def _process_recv_msg(self, face_bbox_per_image_with_audio):
        self.recv_msg_buffer.append(face_bbox_per_image_with_audio)
        self.recv_counter += 1
        if self.recv_counter == self.per_seq_recv_times:
            self.recv_msg_buffer_copy = copy.deepcopy(self.recv_msg_buffer)
            self.seq_id = (self.recv_msg_buffer_copy[0]).seq_id
            self.do_ASD_flag = True
            self.recv_counter = 0

    def _get_face_track_result(self):
        track_results = {}
        for bbox_xyxy_and_id in (self.recv_msg_buffer_copy[-1]).bboxes_xyxy_and_ids:
            track_results[bbox_xyxy_and_id.track_id] = {
                "index": [
                    -1,
                ],
                "bbox": [
                    bbox_xyxy_and_id.bbox_xyxy,
                ],
            }
        self.audio = self.recv_msg_buffer_copy[-1].audio
        track_results_keys = list(track_results.keys())
        for index, recv_msg in enumerate(
            reversed(list(self.recv_msg_buffer_copy)[:-1])
        ):
            index = -index - 2
            self.audio = np.concatenate([recv_msg.audio, self.audio])
            for bbox_xyxy_and_id in recv_msg.bboxes_xyxy_and_ids:
                if bbox_xyxy_and_id.track_id in track_results_keys:
                    track_results[bbox_xyxy_and_id.track_id]["index"].insert(0, index)
                    track_results[bbox_xyxy_and_id.track_id]["bbox"].insert(
                        0, bbox_xyxy_and_id.bbox_xyxy
                    )
        face_track_results = []
        for key in track_results_keys:
            face_track_result = {"track_id": key}
            face_track_result["index"] = track_results[key]["index"]
            face_track_result["bbox"] = track_results[key]["bbox"]
            face_track_results.append(face_track_result)
        return face_track_results

    def _crop_face_and_audio(self, track_results_multiple_people):
        cropped_face_images_and_audios = []
        visualization_bboxes = []
        for track_results_single_person in track_results_multiple_people:
            smoothed_track_results = {"x": [], "y": [], "s": []}
            for track_result in track_results_single_person["bbox"]:
                smoothed_track_results["s"].append(
                    max(
                        (track_result[3] - track_result[1]),
                        (track_result[2] - track_result[0]),
                    )
                    / 2
                )
                smoothed_track_results["y"].append(
                    (track_result[1] + track_result[3]) / 2
                )  # crop center y
                smoothed_track_results["x"].append(
                    (track_result[0] + track_result[2]) / 2
                )  # crop center x
            # smoothed_track_results["s"] = scipy.signal.medfilt(
            #     smoothed_track_results["s"], kernel_size=13
            # )  # smooth detections
            # smoothed_track_results["y"] = scipy.signal.medfilt(
            #     smoothed_track_results["y"], kernel_size=13
            # )  # smooth detections
            # smoothed_track_results["x"] = scipy.signal.medfilt(
            #     smoothed_track_results["x"], kernel_size=13
            # )  # smooth detections
            face_images = []
            audio_recvs = None
            for i, index in enumerate(track_results_single_person["index"]):
                cs = self.algs_args.crop_scale
                bs = smoothed_track_results["s"][i]  # Detection box size
                bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
                image = self.bridge.imgmsg_to_cv2(
                    (self.recv_msg_buffer_copy[index]).color_image,
                    desired_encoding="bgr8",
                )
                image_padded = np.pad(
                    image,
                    ((bsi, bsi), (bsi, bsi), (0, 0)),
                    "constant",
                    constant_values=(110, 110),
                )
                my = smoothed_track_results["y"][i] + bsi  # BBox center Y
                mx = smoothed_track_results["x"][i] + bsi  # BBox center X
                face_image = image_padded[
                    int(my - bs) : int(my + bs * (1 + 2 * cs)),
                    int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
                ]
                # cv2.error: OpenCV(4.10.0) /io/opencv/modules/imgproc/src/resize.cpp:4152: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
                face_image = cv2.resize(face_image, (224, 224))
                face_images.append(face_image)
                audio_recv = (self.recv_msg_buffer_copy[index]).audio
                if audio_recvs is None:
                    audio_recvs = audio_recv
                else:
                    audio_recvs = np.concatenate([audio_recvs, audio_recv])
            cropped_face_images_and_audios.append(
                {
                    "track_id": track_results_single_person["track_id"],
                    "image": face_images,
                    "audio": audio_recvs,
                }
            )
            # 最后一张图片就是当前图片
            visualization_bbox = {
                "x": smoothed_track_results["x"][-1],
                "y": smoothed_track_results["y"][-1],
                "s": smoothed_track_results["s"][-1],
            }
            visualization_bboxes.append(visualization_bbox)
        return cropped_face_images_and_audios, visualization_bboxes

    def _process_audio_msg(self, audio_recv):
        audio_feature = python_speech_features.mfcc(
            audio_recv,
            16000,
            numcep=13,
            winlen=0.025 * 25 / self.audio_image_pub_frequency,
            winstep=0.010 * 25 / self.audio_image_pub_frequency,
        )
        return audio_feature

    def _detect_active_speaker(self, images_and_audios_multiple_people):
        scores_multiple_people = []
        for images_and_audios_one_person in images_and_audios_multiple_people:
            sequence_length = len(images_and_audios_one_person["image"])
            images_one_person = images_and_audios_one_person["image"]
            images_detect = []
            for face_image in images_one_person:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = cv2.resize(face_image, (224, 224))
                face_image = face_image[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                images_detect.append(face_image)
            images_detect = np.array(images_detect)
            audios_one_person = images_and_audios_one_person["audio"]  # np.array
            audios_detect = self._process_audio_msg(audios_one_person)  # np.array
            with torch.no_grad():
                audios_detect = torch.FloatTensor(audios_detect).unsqueeze(0).cuda()
                images_detect = torch.FloatTensor(images_detect).unsqueeze(0).cuda()
                audio_embeddings = (
                    self.active_speaker_detector.model.forward_audio_frontend(
                        audios_detect
                    )
                )
                visual_embeddings = (
                    self.active_speaker_detector.model.forward_visual_frontend(
                        images_detect
                    )
                )
                audio_embeddings, visual_embeddings = (
                    self.active_speaker_detector.model.forward_cross_attention(
                        audio_embeddings, visual_embeddings
                    )
                )
                out = self.active_speaker_detector.model.forward_audio_visual_backend(
                    audio_embeddings, visual_embeddings
                )
                score = self.active_speaker_detector.lossAV.forward(out, labels=None)
                # 暂时先不做平滑,如果数据变动太剧烈再做平滑
                # 看了talk-net源码，与label求损失函数也用的网络推理结果，没有做平滑
                # # smooth score(according to original repo's visualization func)
                # score = float(
                #     np.mean(score[max(0, sequence_length - smooth_window_size) :])
                # )
                # # scores_multiple_people.append(current_face_score)

                score = np.count_nonzero(score > 0)
                scores_multiple_people.append(
                    {
                        "tracker_id": images_and_audios_one_person["track_id"],
                        "score": score,
                    }
                )
        return scores_multiple_people

    def _viz_draw_bbox_and_text_on_image(
        self, scores_multiple_people, visualization_bboxes, image_show
    ):
        for score, bbox in zip(scores_multiple_people, visualization_bboxes):
            color = 255 if score >= 0 else 0
            score = round(score, 1)
            cv2.rectangle(
                image_show,
                (int(bbox["x"] - bbox["s"]), int(bbox["y"] - bbox["s"])),
                (int(bbox["x"] + bbox["s"]), int(bbox["y"] + bbox["s"])),
                (0, color, 255 - color),
                8,
            )
            cv2.putText(
                image_show,
                str(score),
                (int(bbox["x"] - bbox["s"]), int(bbox["y"] - bbox["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, color, 255 - color),
                5,
            )

    def _visualization(self, max_times_track_id, face_track_results):
        current_image = self.bridge.imgmsg_to_cv2(
            (self.recv_msg_buffer_copy[-1]).color_image,
            desired_encoding="bgr8",
        )
        for face_track_result in face_track_results:
            track_id = face_track_result["track_id"]
            bbox_color = None
            if track_id == max_times_track_id:
                self.count += 1
                print(self.count)
                bbox_color = (0, 0, 255)
            else:
                bbox_color = (0, 255, 0)
            x1, y1, x2, y2 = face_track_result["bbox"][-1]
            cv2.rectangle(
                current_image,
                (x1, y1),
                (x2, y2),
                bbox_color,
                thickness=2,
            )
        cv2.imshow("ASD", current_image)
        cv2.waitKey(1)

    def _pub_ASD_result(self, scores_multiple_people):
        max_times_track_id = -1
        max_times = -1
        # score大于0的max_times有多个人，先暂时取第一个人，在场景假设为轮流说话，就算可能存在几帧误检按理说不会有这种情况
        for track_id_and_score in scores_multiple_people:
            if track_id_and_score["score"] < self.algs_args.min_ASD_greater_than_zero:
                continue
            if track_id_and_score["score"] > max_times_track_id:
                max_times = track_id_and_score["score"]
                max_times_track_id = track_id_and_score["tracker_id"]
        if max_times_track_id != -1:
            pub_msg = ActiveSpeakerAudio()
            pub_msg.seq_id = self.seq_id
            pub_msg.track_id = max_times_track_id
            pub_msg.audio = self.audio.tolist()
            self.pub_ASD_result.publish(pub_msg)
        else:
            self._pub_terminate_ASR_signal()

        if self.viz_flag:
            return max_times_track_id

    def _pub_terminate_ASR_signal(self):
        pub_msg = ActiveSpeakerAudio()
        pub_msg.seq_id = self.seq_id
        pub_msg.track_id = -1
        self.pub_ASD_result.publish(pub_msg)


def main():
    rospy.init_node("active_speaker_detection")
    ASD_detector = ASDetector()
    try:
        rospy.spin()
    finally:
        ASD_detector.thread_shutdown()


if __name__ == "__main__":
    main()
