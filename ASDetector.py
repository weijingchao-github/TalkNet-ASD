# fps:8.5
import os
import sys

path = os.path.dirname(__file__)
sys.path.insert(0, path)

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
from model.faceDetector.s3fd import S3FD
from sensor_msgs.msg import CameraInfo
from talkNet import talkNet as TalkNet

from active_speaker_detection.msg import (
    ActiveSpeakerPositionArrayCamera3D,
    Camera3DPosition,
)
from sync_perception_signal.msg import VisionAudioPair


class ASDetector:
    def __init__(self):
        # 接收到的一定是对齐的声音&视频信号对
        # algs' parameters
        self.algs_args = SimpleNamespace(
            iou_thres=0.5,
            face_det_scale=0.25,
            buffer_max_length=25,
            num_failed_det=2,
            min_track=1,
            min_face_size=1,
            crop_scale=0.40,
        )

        # Model init
        self.face_detector = S3FD(device="cuda")
        self.active_speaker_detector = TalkNet()
        self.active_speaker_detector.loadParameters(
            os.path.join(os.path.dirname(__file__), "pretrain_TalkSet.model")
        )
        self.active_speaker_detector.eval()

        # ROS init
        rospy.Subscriber(
            "/sync_perception/sync_vision_audio_signal",
            VisionAudioPair,
            self._process_perception_msg,
            queue_size=10,
        )
        self.pub_active_speaker_position_array_camera_3D = rospy.Publisher(
            "active_speaker_position_array_camera_3D",
            ActiveSpeakerPositionArrayCamera3D,
            queue_size=10,
        )
        self.bridge = CvBridge()
        self.perception_msg_buffer = deque(maxlen=self.algs_args.buffer_max_length)
        self.depth_info = rospy.wait_for_message(
            "/camera/depth/camera_info", CameraInfo
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
            if len(self.perception_msg_buffer) < self.algs_args.buffer_max_length:
                # 从python GIL里抢占用
                time.sleep(0.1)
                continue

            current_perception_msg_buffer = copy.deepcopy(
                list(self.perception_msg_buffer)
            )
            current_image = {
                "color_image": current_perception_msg_buffer[-1]["visual"][
                    "color_image"
                ],
                "depth_image": current_perception_msg_buffer[-1]["visual"][
                    "depth_image"
                ],
            }

            # 如果最新的一帧里没有人脸，那就不进行ASD
            if len(current_perception_msg_buffer[-1]["visual"]["bboxes_position"]) == 0:
                cv2.imshow("image", current_image["color_image"])
                cv2.waitKey(1)
                continue

            # track faces
            track_results = self._track_face(current_perception_msg_buffer)
            if (
                len(track_results) == 0
            ):  # len(track_result) >= self.algs_args.min_track这一句没通过
                cv2.imshow("image", current_image["color_image"])
                cv2.waitKey(1)
                continue

            # crop faces and audios
            cropped_face_images_and_audios, visualization_bboxes = (
                self._crop_face_and_audio(track_results, current_perception_msg_buffer)
            )

            # detect active speaker
            scores_multiple_people = self._detect_active_speaker(
                cropped_face_images_and_audios
            )

            # visualization
            self._viz_draw_bbox_and_text_on_image(
                scores_multiple_people,
                visualization_bboxes,
                current_image["color_image"],
            )
            cv2.imshow("image", current_image["color_image"])
            cv2.waitKey(1)

            # Compute and pub active speaker's position in Camera 3D Coordinates
            self._compute_and_pub_active_speaker_position(
                scores_multiple_people,
                visualization_bboxes,
                current_image["depth_image"],
            )

    def _process_perception_msg(self, audio_visual_pair_msg: VisionAudioPair):
        # time_step = audio_visual_pair_msg.time_step
        audio_recv = np.array(audio_visual_pair_msg.audio, dtype=np.int16)
        color_image_recv = self.bridge.imgmsg_to_cv2(
            audio_visual_pair_msg.color_image, desired_encoding="bgr8"
        )
        depth_image_recv = self.bridge.imgmsg_to_cv2(
            audio_visual_pair_msg.depth_image, desired_encoding="16UC1"
        )  # 深度图的原始编码是16位无符号整数

        # process visual msg
        bboxes_position = self._detect_face(color_image_recv)

        # # process audio msg
        # audio_feature = self._process_audio_msg(audio_recv)

        # # save msgs in buffer
        # perception_msgs = {
        #     # "time_step": time_step,
        #     "visual": {"image": image_recv, "bboxes_position": bboxes_position},
        #     "audio": audio_feature,
        # }

        # save msgs in buffer
        perception_msgs = {
            # "time_step": time_step,
            "visual": {
                "color_image": color_image_recv,
                "depth_image": depth_image_recv,
                "bboxes_position": bboxes_position,
            },
            "audio": audio_recv,
        }
        self.perception_msg_buffer.append(perception_msgs)

    def _detect_face(self, image_recv):
        image_rgb = cv2.cvtColor(image_recv, cv2.COLOR_BGR2RGB)
        bboxes = self.face_detector.detect_faces(
            image_rgb, conf_th=0.8, scales=[self.algs_args.face_det_scale]
        )
        bboxes_position = [(bbox[:-1]).tolist() for bbox in bboxes]
        return bboxes_position

    def _track_face(self, current_perception_msg_buffer):
        # CPU: Face tracking
        perception_msg_buffer = copy.deepcopy(current_perception_msg_buffer)
        track_results = []
        while True:
            track_result = []
            for frame_index, frame_faces in enumerate(
                reversed(perception_msg_buffer)
            ):  # 每一张图片
                frame_index = -frame_index - 1
                if frame_index == -1:
                    if len(frame_faces["visual"]["bboxes_position"]) == 0:
                        break
                for face_bbox in frame_faces["visual"][
                    "bboxes_position"
                ]:  # 每一张图片里的bbox
                    if len(track_result) == 0:
                        track_result.insert(0, (frame_index, face_bbox))
                        frame_faces["visual"]["bboxes_position"].remove(face_bbox)
                    elif (
                        frame_index != track_result[0][0]
                        and abs(frame_index - track_result[0][0])
                        <= self.algs_args.num_failed_det
                    ):
                        iou = self._bb_intersection_over_union(
                            face_bbox, track_result[0][1]
                        )
                        if iou > self.algs_args.iou_thres:
                            track_result.insert(0, (frame_index, face_bbox))
                            frame_faces["visual"]["bboxes_position"].remove(face_bbox)
                    else:
                        break
            if len(track_result) == 0:
                break
            if len(track_result) >= self.algs_args.min_track:
                frame_index = np.array(
                    [
                        track_result_one_piece[0]
                        for track_result_one_piece in track_result
                    ]
                )
                frame_bbox = np.array(
                    [
                        track_result_one_piece[1]
                        for track_result_one_piece in track_result
                    ]
                )
                track_result_indexes = np.arange(frame_index[0], frame_index[-1] + 1)
                track_result_bboxes = []
                for i in range(4):
                    interpfn = scipy.interpolate.interp1d(frame_index, frame_bbox[:, i])
                    track_result_bboxes.append(interpfn(track_result_indexes))
                track_result_bboxes = np.stack(track_result_bboxes, axis=1)
                if (
                    max(
                        np.mean(track_result_bboxes[:, 2] - track_result_bboxes[:, 0]),
                        np.mean(track_result_bboxes[:, 3] - track_result_bboxes[:, 1]),
                    )
                    > self.algs_args.min_face_size
                ):
                    track_results.append(
                        {"index": track_result_indexes, "bbox": track_result_bboxes}
                    )

        return track_results

    # def _crop_face_and_audio(
    #     self, track_results_multiple_people, current_perception_msg_buffer
    # ):
    #     cropped_face_images_and_audios = []
    #     visualization_bboxes = []
    #     for track_results_single_person in track_results_multiple_people:
    #         smoothed_track_results = {"x": [], "y": [], "s": []}
    #         for track_result in track_results_single_person["bbox"]:
    #             smoothed_track_results["s"].append(
    #                 max(
    #                     (track_result[3] - track_result[1]),
    #                     (track_result[2] - track_result[0]),
    #                 )
    #                 / 2
    #             )
    #             smoothed_track_results["y"].append(
    #                 (track_result[1] + track_result[3]) / 2
    #             )  # crop center y
    #             smoothed_track_results["x"].append(
    #                 (track_result[0] + track_result[2]) / 2
    #             )  # crop center x
    #         smoothed_track_results["s"] = scipy.signal.medfilt(
    #             smoothed_track_results["s"], kernel_size=13
    #         )  # smooth detections
    #         smoothed_track_results["y"] = scipy.signal.medfilt(
    #             smoothed_track_results["y"], kernel_size=13
    #         )  # smooth detections
    #         smoothed_track_results["x"] = scipy.signal.medfilt(
    #             smoothed_track_results["x"], kernel_size=13
    #         )  # smooth detections
    #         face_images = []
    #         audio_features = None
    #         for i, image_index in enumerate(track_results_single_person["index"]):
    #             cs = self.algs_args.crop_scale
    #             bs = smoothed_track_results["s"][i]  # Detection box size
    #             bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
    #             image = current_perception_msg_buffer[image_index]["visual"]["image"]
    #             image_padded = np.pad(
    #                 image,
    #                 ((bsi, bsi), (bsi, bsi), (0, 0)),
    #                 "constant",
    #                 constant_values=(110, 110),
    #             )
    #             my = smoothed_track_results["y"][i] + bsi  # BBox center Y
    #             mx = smoothed_track_results["x"][i] + bsi  # BBox center X
    #             face_image = image_padded[
    #                 int(my - bs) : int(my + bs * (1 + 2 * cs)),
    #                 int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
    #             ]
    #             face_image = cv2.resize(face_image, (224, 224))
    #             face_images.append(face_image)
    #             audio_feature = current_perception_msg_buffer[image_index]["audio"]
    #             if audio_features is None:
    #                 audio_features = audio_feature
    #             else:
    #                 audio_features = np.concatenate(
    #                     [audio_features, audio_feature], axis=0
    #                 )
    #         cropped_face_images_and_audios.append(
    #             {"image": face_images, "audio": audio_features}
    #         )
    #         # 最后一张图片就是当前图片
    #         visualization_bboxes.append(smoothed_track_results)
    #     return cropped_face_images_and_audios, visualization_bboxes

    def _crop_face_and_audio(
        self, track_results_multiple_people, current_perception_msg_buffer
    ):
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
            for i, image_index in enumerate(track_results_single_person["index"]):
                cs = self.algs_args.crop_scale
                bs = smoothed_track_results["s"][i]  # Detection box size
                bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
                image = current_perception_msg_buffer[image_index]["visual"][
                    "color_image"
                ]
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
                audio_recv = current_perception_msg_buffer[image_index]["audio"]
                if audio_recvs is None:
                    audio_recvs = audio_recv
                else:
                    audio_recvs = np.concatenate([audio_recvs, audio_recv])
            cropped_face_images_and_audios.append(
                {"image": face_images, "audio": audio_recvs}
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
            audio_recv, 16000, numcep=13, winlen=0.025, winstep=0.010
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
            # length = min(
            #     (audios_detect.shape[0] - audios_detect.shape[0] % 4) / 100,
            #     images_detect.shape[0] / 25,
            # )
            # audios_detect = audios_detect[audios_detect.shape[0] - int(round(length )), :]
            # assert (
            #     audios_detect.shape[0] - audios_detect.shape[0] % 4
            # ) / 100 == images_detect.shape[0] / 25

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
                # smooth score(according to original repo's visualization func)
                smooth_window_size = 10
                current_face_score = float(
                    np.mean(score[max(0, sequence_length - smooth_window_size) :])
                )
                scores_multiple_people.append(current_face_score)
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

    def _compute_and_pub_active_speaker_position(
        self, scores_multiple_people, bboxes, depth_image
    ):
        # depth intrin
        fx = self.depth_info.K[0]
        fy = self.depth_info.K[4]
        cx = self.depth_info.K[2]
        cy = self.depth_info.K[5]

        active_speaker_position_array_camera_3D = ActiveSpeakerPositionArrayCamera3D()
        for score, bbox in zip(scores_multiple_people, bboxes):
            if score <= 0:
                continue
            active_speaker_position_camera_3D = Camera3DPosition()
            x_2D, y_2D = int(bbox["x"]), int(bbox["y"])
            print(f"{x_2D}, {y_2D}")
            z_camera_3D = depth_image[y_2D, x_2D] * 0.001  # m
            if z_camera_3D == 0:
                print("Camera Error!")
                raise Exception("Camera error!")
            x_camera_3D = (x_2D - cx) * z_camera_3D / fx
            y_camera_3D = (y_2D - cy) * z_camera_3D / fy
            active_speaker_position_camera_3D.x = x_camera_3D
            active_speaker_position_camera_3D.y = y_camera_3D
            active_speaker_position_camera_3D.z = z_camera_3D
            active_speaker_position_array_camera_3D.data.append(
                active_speaker_position_camera_3D
            )
        if len(active_speaker_position_array_camera_3D.data) > 0:
            self.pub_active_speaker_position_array_camera_3D.publish(
                active_speaker_position_array_camera_3D
            )

    def _bb_intersection_over_union(self, boxA, boxB, evalCol=False):
        # CPU: IOU Function to calculate overlap between two image
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if evalCol == True:
            iou = interArea / float(boxAArea)
        else:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


def main():
    rospy.init_node("active_speaker_detection")
    asd_detector = ASDetector()
    try:
        rospy.spin()
    finally:
        asd_detector.thread_shutdown()


if __name__ == "__main__":
    main()
