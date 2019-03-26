#!/usr/bin/env python
# -*- Codigin: UTF-8 -*-
"""unit test form layers decorator."""
import unittest
import os

import cv2
from blockmatching import *
import matplotlib.pyplot as plt


CDIR = os.path.dirname(os.path.abspath(__file__))

VIDEOTEST = CDIR + '/videos/car.mp4'

@dlayers(0.01, 19, 19, 7)
def background(videofile):
    cap = cv2.VideoCapture(videofile)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            break
        yield frame

for bg, meand, layers in background(VIDEOTEST):
    fg = cv2.resize(bg, (0,0), fx=0.5, fy=0.5)
    if layers:
        cv2.imshow("Layer 0", layers[0])

    cv2.imshow('Background', fg)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

'''
first_frame = True
background = None
old_frame = None
alpha = 0.01
width = 3
height = 3
cap = cv2.VideoCapture(VIDEOTEST)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    if first_frame is True:
        background = BackgroundSubtractor(alpha, frame)
        old_frame = background.foreground(frame)
        first_frame = False
    else:
        foreground = background.foreground(frame)

        XP, YP, XD, YD = block_matching(old_frame,
                                        foreground,
                                        width,
                                        height)
        U, V, object_tops, meand = clustering(XD, YD, XP, YP)
        old_frame = foreground.copy()

    fg = cv2.resize(background.background, (0,0), fx=0.5, fy=0.5)


    cv2.imshow('Background', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
'''
