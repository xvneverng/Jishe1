import math
import threading
import pythoncom
from collections import deque

import numpy as np
import pyttsx3

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from pykalman import KalmanFilter
import torch
import cv2
engine = pyttsx3.init()
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

'''
def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['person']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image

'''


def play_warning():
    engine.say("小心碰撞")
    engine.runAndWait()

def plot_bboxes(image, bboxes, line_thickness=None,played_warnings=None):
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    for (x1, y1, x2, y2, cls_id, pos_id, speed, distance, risk_level) in bboxes:
        if cls_id in ['person']:
            color = (0, 0, 255)
        else:
            if risk_level == "high"and distance < 10:
                color = (0, 0, 255)
                if pos_id not in played_warnings :
                    warning_thread = threading.Thread(target=play_warning)
                    warning_thread.start()
                    played_warnings[pos_id] = True
            elif risk_level == "medium":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)

        labels = [
            (f"{distance:.2f}m", (255, 0, 255)),
            (f"{speed:.2f}m/s", (0, 255, 0)),
            (f"ID-{pos_id}", (255, 255, 255)),
        ]

        y_offset = c1[1] - 2
        font_scale = 0.5 * tl / 2
        for label, text_color in labels:
            t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=2)[0]
            text_position = (c1[0], y_offset)
            cv2.putText(image, label, text_position, 0, font_scale, text_color, thickness=2, lineType=cv2.LINE_AA)
            y_offset -= t_size[1] + 3

    return image


import numpy as np
from pykalman import KalmanFilter

class SpeedKalmanFilter:
    def __init__(self, initial_speed=0.0):
        self.kf = KalmanFilter(transition_matrices=[[1, 1], [0, 1]], observation_matrices=[[1, 0]])

        # 设置过程噪声协方差矩阵
        process_noise = 1e-5
        self.kf.transition_covariance = np.array([[process_noise, 0],
                                                  [0, process_noise]])

        # 设置观测噪声协方差矩阵
        observation_noise = 1e-4
        self.kf.observation_covariance = np.array([[observation_noise]])

    def update(self, speed_measurement):
        if not isinstance(speed_measurement, list):
            speed_measurement = [speed_measurement]

        if len(speed_measurement) > 1:
            self.kf = self.kf.em(speed_measurement, n_iter=5)
            (filtered_state_means, _) = self.kf.filter(speed_measurement)
            filtered_speed = filtered_state_means[-1, 1]
        else:
            filtered_speed = speed_measurement[0]

        return filtered_speed

played_warnings = {}
def update_tracker(target_detector, image):
    FOCAL_LENGTH = 700  # 焦距，单位：像素
    CAMERA_HEIGHT = 2  # 摄像头高度，单位：米
    PIXEL_TO_METERS = 0.2  # 像素到米的转换系数（根据您的摄像头实际参数进行修改）
    real_hight_car = 59.08  # 汽车高度
    real_hight_motorcycle = 47.24  # 摩托车高度
    real_hight_bus = 125.98  # 公交车高度
    real_hight_truck = 137.79  # 卡车高度



    new_faces = []
    _, bboxes = target_detector.detect(image)
    speed_kalman_filters = {}
    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:
        obj = [
            int((x1 + x2) / 2), int((y1 + y2) / 2),
            x2 - x1, y2 - y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        if cls_ == "car" or cls_ == "truck" or cls_ == "bus":
            width = x2 - x1
            #distance = FOCAL_LENGTH * CAMERA_HEIGHT / (width * PIXEL_TO_METERS)
            dis_inch = (real_hight_car * FOCAL_LENGTH) /width*2
            dis_cm = dis_inch * 2.54
            dis_cm = int(dis_cm)
            dis_m = dis_cm / 100
            if track_id in target_detector.prev_car_positions:
                prev_x1, prev_y1, _, _ = target_detector.prev_car_positions[track_id]
                delta_x = abs(x1 - prev_x1) * PIXEL_TO_METERS
                delta_y = abs(y1-prev_x1)
                delta_t = 1 / target_detector.frame_rate
                speed = [math.sqrt((delta_x**2+delta_y**2))/(abs(x2-x1))*delta_t/PIXEL_TO_METERS*10]
            else:
                speed = [0]

            if track_id not in speed_kalman_filters:
                speed_kalman_filters[track_id] = SpeedKalmanFilter(initial_speed=0.0)

            filtered_speed = speed_kalman_filters[track_id].update(speed)
            time_to_collision = dis_m / filtered_speed if speed[0] > 0 else float("inf")

            if time_to_collision < 5:
                risk_level = "high"
            elif time_to_collision < 3:
                risk_level = "medium"
            else:
                risk_level = "low"

            target_detector.car_distances[track_id] = dis_m
            target_detector.car_speeds[track_id] = speed
            target_detector.car_collision_risk[track_id] = risk_level
            target_detector.prev_car_positions[track_id] = (x1, y1, x2, y2)
            #print("x1:",x1,"x2:",x2)
            x_center=(x1+x2)/2
            y_center=(y1+y2)/2

            bboxes2draw.append((x1, y1, x2, y2, cls_, track_id, filtered_speed, dis_m, risk_level))


        if cls_ == 'face':
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )

    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw, played_warnings=played_warnings)

    return image, new_faces, face_bboxes
