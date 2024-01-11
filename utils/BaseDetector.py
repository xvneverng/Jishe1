from tracker import update_tracker
import cv2


class baseDet(object):

    def __init__(self):

        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.car_distances = {}  # 存储车辆间距离
        self.car_speeds = {}  # 存储车辆相对速度
        self.car_collision_risk = {}  # 存储碰撞风险等级
        self.prev_car_positions = {}  # 存储车辆上一帧的位置
        self.frame_rate = 24  # 视频帧率（根据您的视频实际帧率进行修改）

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im):
        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': [],
            'car_distances': self.car_distances,
            'car_speeds': self.car_speeds,
            'car_collision_risk': self.car_collision_risk
        }

        self.frameCounter += 1

        im, faces, face_bboxes = update_tracker(self, im)

        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
