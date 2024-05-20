# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1500, 720)
        self.video_window = QtWidgets.QLabel(Form)
        self.video_window.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        font = QtGui.QFont()
        font.setPointSize(22)
        self.video_window.setFont(font)
        self.video_window.setFrameShape(QtWidgets.QFrame.Box)
        self.video_window.setFrameShadow(QtWidgets.QFrame.Plain)
        self.video_window.setObjectName("video_window")
        # 垂直布局
        self.verticalLayoutWidget = QtWidgets.QWidget(Form)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(1310, 280, 181, 231))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        # 原始
        self.origin_camera = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.origin_camera.setFont(font)
        self.origin_camera.setObjectName("origin_camera")
        self.verticalLayout.addWidget(self.origin_camera)
        # 虚拟背景
        self.virtualbackground_camera = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.virtualbackground_camera.setFont(font)
        self.virtualbackground_camera.setObjectName("virtualbackground_camera")
        self.verticalLayout.addWidget(self.virtualbackground_camera)
        # 面具
        self.mask_camera = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.mask_camera.setFont(font)
        self.mask_camera.setObjectName("mask_camera")
        self.verticalLayout.addWidget(self.mask_camera)
        # 背景+面具
        self.all_camera = QtWidgets.QPushButton(self.verticalLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.all_camera.setFont(font)
        self.all_camera.setObjectName("all_camera")
        self.verticalLayout.addWidget(self.all_camera)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "许子强"))
        self.video_window.setText(_translate("Form", " "))
        self.origin_camera.setText(_translate("Form", "开启摄像头"))
        self.virtualbackground_camera.setText(_translate("Form", "虚拟背景"))
        self.mask_camera.setText(_translate("Form", "面具特效"))
        self.all_camera.setText(_translate("Form", "背景+面具"))
