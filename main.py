# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:43:14 2019

@author: Teng
"""
import sys
import os.path as osp
#path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
#if path not in sys.path:
#    sys.path.append(path)

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

from gui.dialog import Ui_Transform
from gui.transformation import transform
from test import create_models, defect_inspection

img_height = 640
img_width = 960
conf_thresh = 0.5
classes1 = ('background', 'insulator', 'nuts')
classes2 = ('normal', 'broken', 'normal', 'lost', 'loose', 'obscured')
colors1 = [[0.5, 0.5, 0.5, 1],
           [0.1, 0.7, 0.2, 1],
           [0.1, 0.2, 0.7, 1]]

colors2 = [[0.0, 0.5, 1.0, 1],
          [0.9, 0.1, 0, 1],
          [0.0, 1.0, 0.6, 1],
          [0.82, 1, 0, 1],
          [0, 0.7, 1, 1],
          [0.7, 0.7, 0.7, 1]]



class Detector(QWidget):
    def __init__(self):
        super(Detector, self).__init__()

        self.resize(1600, 875)
        self.setWindowTitle("4C Detector")

        self.label = QLabel(self)
        self.label.setFixedSize(1200, 800)
        self.label.move(100, 30)

        self.label.setStyleSheet("QLabel{background:white;}"
                                 "QLabel{color:rgb(240,240,240,120);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )
        
        self.ui = Ui_Transform()
        self.dlg = QtWidgets.QDialog()
        
        self.models = None
        self.loc = None

        btn = QPushButton(self)
        btn.setText("打开")
        btn.move(1400, 30)
        btn.clicked.connect(self.openimage)
        
        btn1 = QPushButton(self)
        btn1.setText("载入模型")
        btn1.move(1400, 100)
        btn1.clicked.connect(self.load_model)
        
        btn2 = QPushButton(self)
        btn2.setText("定位")
        btn2.move(1400, 170)
        btn2.clicked.connect(self.locate)
        
        btn3 = QPushButton(self)
        btn3.setText("检测")
        btn3.move(1400, 240)
        btn3.clicked.connect(self.inspect)
        
        btn3 = QPushButton(self)
        btn3.setText("变换")
        btn3.move(1400, 310)
        btn3.clicked.connect(self.transform)
        
        btn3 = QPushButton(self)
        btn3.setText("重置")
        btn3.move(1400, 380)
        btn3.clicked.connect(self.reset)
        
    def openimage(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "Open an image", "", "*.jpg;;*.png;;All Files(*)")
        print(self.imgName)
        self.image = cv2.imread(self.imgName)
        show = self.convert2pixmap(self.image)
        self.label.setPixmap(show)
        
        
    def convert2pixmap(self, img):
        width, height = self.label.width(), self.label.height()
        pixmap = cv2.resize(img, (width, height))
        pixmap = QImage(pixmap, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(pixmap)
        return pixmap

        
    def reset(self):
        self.image = cv2.imread(self.imgName)
        show = self.convert2pixmap(self.image)
        self.label.setPixmap(show)        
        
    def transform(self):
        self.ui.setupUi(self.dlg)
        if self.dlg.exec():
            print('ok')
            scale = self.ui.ScaleSpinBox.value()
            offset = self.ui.TranslateSpinBox.value()
            angle = self.ui.RotatespinBox.value()
            brightness = self.ui.BrightspinBox.value()
            contrast = self.ui.ContrastSpinBox.value()
            gamma = self.ui.GammaSpinBox.value()
            print(scale, '\n', offset, '\n', angle, '\n',
                  brightness, '\n', contrast, '\n', gamma)
            
            self.image = transform(self.image, scale, offset, angle,
                                   brightness, contrast, gamma)
            show = self.convert2pixmap(self.image)
            self.label.setPixmap(show)

        
    def load_model(self):
        self.models = create_models()
        
    def locate(self):
        assert self.models != None, 'Please load model first !'
        assert self.image.size > 0, 'empty image !'
        img = cv2.resize(self.image, (img_width, img_height))
        img = np.expand_dims(img, 0)
        img = img.astype('float32')
        self.loc = self.models[0].predict(img)[0]
        scores = self.loc[:, 1]
        inds = np.where(scores > conf_thresh)[0]
        self.loc = np.expand_dims(self.loc[inds], 0)
        # 将预测的框映射到原图上
        self.loc[:, :, (2, 4)] *= float(self.image.shape[1]) / img_width
        self.loc[:, :, (3, 5)] *= float(self.image.shape[0]) / img_height
        self.draw_boxes(self.loc, classes1, colors1)
        
        img = QPixmap('cache/cache.jpg').scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img)
        
        
    def inspect(self):
        assert not self.loc is None, 'Please run locate first !'
        res = defect_inspection(self.models[1], self.models[2], self.image, self.loc)
        self.draw_boxes(res, classes2, colors2)
        
        img = QPixmap('cache/cache.jpg').scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img) 
        
    def draw_boxes(self, y_pred, classes, colors):
        img = self.image.copy()
        fig = plt.figure()
        fig.set_size_inches(img.shape[1] / 100, img.shape[0] / 100)
        linewidth = 2 * img.shape[1] / 1200
        font_size = linewidth * 7
        plt.imshow(img)
    
        current_axis = plt.gca()
    
        for box in y_pred[0]:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=linewidth))  
            current_axis.text(xmin, ymin, label, size=font_size, color='white', bbox={'facecolor':color, 'alpha':1})
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)
        plt.savefig('cache/cache.jpg', quality=95)
        plt.close()
        


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    d = Detector()
    d.show()
    sys.exit(app.exec_())
    
    
    