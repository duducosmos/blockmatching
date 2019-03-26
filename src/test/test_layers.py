#!/usr/bin/env python
# -*- Codigin: UTF-8 -*-
"""unit test form layers decorator."""
import unittest
import os

import cv2
from blockmatching import layers

CDIR = os.path.dirname(os.path.abspath(__file__))

VIDEOTEST = CDIR + '/videos/car.mp4'

class TestLayers(unittest.TestCase):
    "Test layers decorator."

    def test_background(self):
        "Background subtaction."

        @layers
        def background(videofile, alpha, width, height):
            print(videofile)
            cap = cv2.VideoCapture(videofile)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    break
                yield frame

        bcks = background(VIDEOTEST, alpha=0.01, width=3, height=3)
        for b in bcks:
            fg = cv2.resize(b, (0,0), fx=0.5, fy=0.5)
            cv2.imshow('Background', b)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    unittest.main()
