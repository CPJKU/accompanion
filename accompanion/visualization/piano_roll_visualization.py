import cv2
import os

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


SCORE_COLOR = [105, 105, 105, 255]

PERFORMANCE_COLOR = [8, 104, 172, 255]
ACCOMPANIMENT_COLOR = [217, 95, 14, 255]

# PERFORMANCE_COLOR = [44, 162, 95, 255]
# ACCOMPANIMENT_COLOR = [227, 74, 51, 255]


class PianoRollVisualization(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):

        super(PianoRollVisualization, self).__init__(*args, **kwargs)

        self.frame_pxls = 4
        self.pitch_pixls = 3

        self.max_frames = 1000

        self.piano_frame = QtWidgets.QLabel()
        self.pr_frame = QtWidgets.QLabel()
        self.piano_frame.setStyleSheet("border : 2px solid black")
        self.pr_frame.setStyleSheet("border : 2px solid black")

        self.h = 350
        pixmap = QtGui.QPixmap(os.path.join(CURRENT_DIR, "assets", "piano.png"))
        self.piano_frame.setFixedSize(80, self.h)
        self.pr_frame.setFixedHeight(self.h)
        self.piano_frame.setPixmap(pixmap.scaled(self.piano_frame.width(), self.h, QtCore.Qt.IgnoreAspectRatio))

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.piano_frame)
        layout.addWidget(self.pr_frame)

        self.setLayout(layout)

    def update_pr(self, pr, pred_idx=None, midi_color=None):

        if len(pr.shape) == 3:

            imgs = []
            for i in range(pr.shape[0]):

                img = np.zeros(pr[i].shape + (4,), dtype=np.uint8)

                if midi_color is not None:
                    for idx, c in enumerate(midi_color[i]):
                        img[..., idx][img[..., idx] == 0] = c

                img[..., -1][pr[i] > 0] = (255*(pr[i]/127)).astype(np.uint8)[pr[i] > 0]
                img[pr[i] == 0] = 0
                imgs.append(img)

            img = cv2.addWeighted(imgs[0], 1., imgs[1], 1., 0)

        else:
            # plot score
            img = cv2.cvtColor(pr, cv2.COLOR_GRAY2RGBA)
            if midi_color is not None:
                for idx, c in enumerate(midi_color):
                    img[..., idx][img[..., idx] == 0] = c

        # plot prediction
        if pred_idx is not None:
            cv2.line(img, (pred_idx, 0), (pred_idx, img.shape[1]), (255, 0, 0, 255), thickness=4)

        img = QtGui.QImage(img, img.shape[1], img.shape[0],  QtGui.QImage.Format_RGBA8888)

        self.pr_frame.setPixmap(QtGui.QPixmap(img).scaled(self.pr_frame.width(), self.h))
