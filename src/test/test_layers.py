#!/usr/bin/env python
# -*- Codigin: UTF-8 -*-
"""unit test form layers decorator."""
import unittest
import os

import cv2
from blockmatching import dlayers

CDIR = os.path.dirname(os.path.abspath(__file__))

VIDEOTEST = CDIR + '/videos/car.mp4'

class TestLayers(unittest.TestCase):
    "Test layers decorator."

    def test_background(self):
        "Background subtaction."
        @dlayers(0.01, 3, 3, 7)
        def background(videofile):
            print(videofile)
            cap = cv2.VideoCapture(videofile)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    break
                yield frame
        for bg, fg, mask, meand, layers in background(VIDEOTEST):
            pass

if __name__ == "__main__":
    unittest.main()
