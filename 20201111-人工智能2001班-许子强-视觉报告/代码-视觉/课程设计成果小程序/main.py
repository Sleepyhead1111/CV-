import sys

import time
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from queue import LifoQueue

import mediapipe as mp
from UI import Ui_Form
from face_func import *  # 已含cv2

Decode2Play = LifoQueue()
# 默认是内置摄像头
cameraIndex = 0


class cvDecode(QThread):
    def __init__(self):
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.cap = cv2.VideoCapture(cameraIndex)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.fps_ = 0

    def run(self):
        # print("当前线程 cvDecode: self.threadFlag:{}".format(self.threadFlag))
        pTime = 0
        # widght = int(self.cap.get(3))  # 在视频流的帧的宽度,3 为编号，不能改
        # height = int(self.cap.get(4))  # 在视频流的帧的高度,4 为编号，不能改
        while self.threadFlag:
            if self.cap.isOpened():
                ret, r1 = self.cap.read()
                if not ret:
                    print("can't grab frame.")
                    break
                img = cv2.flip(r1, 1)
                # 算帧率
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                self.fps_ = fps
                cv2.putText(img, f'fps : {int(fps)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                time.sleep(0.01)
                # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源
                if ret:
                    Decode2Play.put(img)   # 解码后的数据放到队列中
                del img


class play_Work(QThread):  # 在 UI 界面中输出识别后的画面
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playLabel = QLabel()  # 初始化 QLabel 对象

    def run(self):
        while self.threadFlag:
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                while not Decode2Play.empty():
                    Decode2Play.get()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # RGB888——三个通道（8，8，8），即每个通道像素值用8位存储，每个通道像素值都为0-255
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在 QLabel 上展示
            time.sleep(0.001)


class play_Work_VirtualBackGround(QThread):  # 在 UI 界面中输出识别后的画面
    def __init__(self):
        super(play_Work_VirtualBackGround, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playLabel = QLabel()  # 初始化 QLabel 对象

    def run(self):
        while self.threadFlag:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            BG_COLOR = (192, 192, 192)  # 浅灰色
            bg_image = cv2.imread('d2.png')
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                while not Decode2Play.empty():
                    Decode2Play.get()
                '''开始生成虚拟背景'''
                if bg_image is None:
                    bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                bg_image = cv2.resize(src=bg_image, dsize=(frame.shape[1], frame.shape[0]))
                # 转换图片到RGB颜色空间
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = mp_selfie_segmentation.SelfieSegmentation(model_selection=1).process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                frame = np.where(condition, frame, bg_image)
                '''结束'''
                # RGB888——三个通道（8，8，8），即每个通道像素值用8位存储，每个通道像素值都为0-255
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在 QLabel 上展示
            time.sleep(0.001)


class play_Work_Mask(QThread):  # 在 UI 界面中输出识别后的画面
    def __init__(self):
        super(play_Work_Mask, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playLabel = QLabel()  # 初始化 QLabel 对象

    def run(self):
        while self.threadFlag:
            # 调用关键点检测模型, face_mesh用于绘人脸面网
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  # 静态图片设置为False,视频设置为True
                                              max_num_faces=3,  # 能检测的最大人脸数
                                              refine_landmarks=True,
                                              # 是否需要对嘴唇、眼睛、瞳孔的关键点进行定位。True:478个关键点；False:468个人脸关键点,缺少瞳孔等关键点。
                                              min_detection_confidence=0.5,  # 人脸检测的置信度
                                              min_tracking_confidence=0.5)  # 人脸追踪的置信度（检测图像时可以忽略）
            # ROI区域中心关键点
            center = {"beard": 164, "guard": 195, "anime": 195, "anonymous": 195, "frontman": 195, "wjj": 195}
            icon_name = "anime"
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                while not Decode2Play.empty():
                    Decode2Play.get()
                '''开始加面具特效'''
                # 获取关键点
                face_landmarks = get_landmarks(frame, face_mesh)
                # 处理特效
                for landmarks in face_landmarks:
                    effect, w, h = process_effects(landmarks, "icons/" + icon_name + ".png", icon_name)
                    # 确定ROI
                    p = center[icon_name]
                    roi = frame[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h, landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w]
                    if effect.shape[:2] == roi.shape[:2]:
                        # 消除特效图像中的白色背景区域
                        # 第一种方法
                        # swap_non_effcet1(effect, roi, 240)
                        # 第二种方法
                        effect = swap_non_effcet2(effect, roi, 240)
                        # 将处理好的特效添加到人脸图像上
                        frame[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h, landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w] = effect
                '''结束'''
                # RGB888——三个通道（8，8，8），即每个通道像素值用8位存储，每个通道像素值都为0-255
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在 QLabel 上展示
            time.sleep(0.001)


class play_Work_All(QThread):  # 在 UI 界面中输出识别后的画面
    def __init__(self):
        super(play_Work_All, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playLabel = QLabel()  # 初始化 QLabel 对象

    def run(self):
        while self.threadFlag:
            mp_selfie_segmentation = mp.solutions.selfie_segmentation
            BG_COLOR = (192, 192, 192)  # 浅灰色
            bg_image = cv2.imread('d2.png')
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                while not Decode2Play.empty():
                    Decode2Play.get()
                '''开始生成虚拟背景'''
                if bg_image is None:
                    bg_image = np.zeros(frame.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                bg_image = cv2.resize(src=bg_image, dsize=(frame.shape[1], frame.shape[0]))
                # 转换图片到RGB颜色空间
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False
                results = mp_selfie_segmentation.SelfieSegmentation(model_selection=1).process(frame)
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                frame = np.where(condition, frame, bg_image)

                # 调用关键点检测模型, face_mesh用于绘人脸面网
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,  # 静态图片设置为False,视频设置为True
                                                  max_num_faces=3,  # 能检测的最大人脸数
                                                  refine_landmarks=True,
                                                  # 是否需要对嘴唇、眼睛、瞳孔的关键点进行定位。True:478个关键点；False:468个人脸关键点,缺少瞳孔等关键点。
                                                  min_detection_confidence=0.5,  # 人脸检测的置信度
                                                  min_tracking_confidence=0.5)  # 人脸追踪的置信度（检测图像时可以忽略）
                # ROI区域中心关键点
                center = {"beard": 164, "guard": 195, "anime": 195, "anonymous": 195, "frontman": 195, "wjj": 195}
                icon_name = "anime"
                '''开始加面具特效'''
                # 获取关键点
                face_landmarks = get_landmarks(frame, face_mesh)
                # 处理特效
                for landmarks in face_landmarks:
                    effect, w, h = process_effects(landmarks, "icons/" + icon_name + ".png", icon_name)
                    # 确定ROI
                    p = center[icon_name]
                    roi = frame[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h, landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w]
                    if effect.shape[:2] == roi.shape[:2]:
                        # 消除特效图像中的白色背景区域
                        # 第一种方法
                        # swap_non_effcet1(effect, roi, 240)
                        # 第二种方法
                        effect = swap_non_effcet2(effect, roi, 240)
                        # 将处理好的特效添加到人脸图像上
                        frame[landmarks[p][1] - int(h / 2):landmarks[p][1] - int(h / 2) + h, landmarks[p][0] - int(w / 2):landmarks[p][0] - int(w / 2) + w] = effect
                '''结束'''
                # RGB888——三个通道（8，8，8），即每个通道像素值用8位存储，每个通道像素值都为0-255
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在 QLabel 上展示
            time.sleep(0.001)


class NameController(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # 初始化时构建窗口
        self.controller()
        self.playwork = None

    def controller(self):
        self.origin_camera.clicked.connect(lambda: self.origin_video_start())
        self.virtualbackground_camera.clicked.connect(lambda: self.VirtualBackGround_video_start())
        self.mask_camera.clicked.connect(lambda: self.mask_video_start())
        self.all_camera.clicked.connect(lambda: self.all_video_start())

    # 输入文本后按下回车即可进行识别，与点击下方按钮效果相同
    def origin_video_start(self):
        if self.playwork and self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()
        global cameraIndex
        cameraIndex = 0
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()
        self.playwork = play_Work()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.video_window
        self.playwork.start()

    def VirtualBackGround_video_start(self):
        if self.playwork  and self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()
        global cameraIndex
        cameraIndex = 0
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()
        self.playwork = play_Work_VirtualBackGround()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.video_window
        self.playwork.start()

    def mask_video_start(self):
        if self.playwork and self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()
        global cameraIndex
        cameraIndex = 0
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()
        self.playwork = play_Work_Mask()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.video_window
        self.playwork.start()

    def all_video_start(self):
        if self.playwork and self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()
        global cameraIndex
        cameraIndex = 0
        self.decodework = cvDecode()
        self.decodework.threadFlag = 1
        self.decodework.start()
        self.playwork = play_Work_All()
        self.playwork.threadFlag = 1
        self.playwork.playLabel = self.video_window
        self.playwork.start()

    def closeEvent(self, event):
        print("关闭线程")
        # Qt 需要先退出循环才能关闭线程
        if self.decodework.isRunning():
            self.decodework.threadFlag = 0
            self.decodework.quit()
        if self.playwork.isRunning():
            self.playwork.threadFlag = 0
            self.playwork.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 实例化一个 app
    window = NameController()     # 实例化一个窗口
    window.show()                 # 以默认大小显示窗口
    sys.exit(app.exec_())

