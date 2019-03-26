#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
"""
Layer decorator.

Decorated function must be a generator that return frame images.

The function will be return the background, the mean velocity and separaed
layers of \'connected\' moving objects in the video.

Parameters
----------

    :parameter int width: width, in pixels, of search box for block matching
                          algorithm. default 3.
    :parameter int height: height, in pixels, of search box for block matching
                          algorithm. default 3.

    :param float  alpha: The background learning factor, its value should
                       be between 0 and 1. The higher the value, the more quickly
                       your program learns the changes in the background. Therefore,
                       for a static background use a lower value, like 0.001. But if
                       your background has moving trees and stuff, use a higher
                       value, maybe start with 0.01.
                       default: 0.01

Return
------
    :return 2d_array background:
    :return list mean_velocity:
    :return list layers:

Example
-------

>>> import cv2
>>> @layers
>>> def background(videofile, alpha, width, height):
>>>     cap = cv2.VideoCapture(videofile)
>>>     while cap.isOpened():
>>>         ret, frame = cap.read()
>>>         if ret == True:
>>>             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
>>>             yield frame
>>> background('/videos/car.mp4', alpha=0.01, width=3, height=3)
"""


from .blockmatching import block_matching
from .background import BackgroundSubtractor
from .clustering import clustering
import cv2


class layers:
    '''
    Layer decorator.
    Parameters
    ----------
    :parameter int width: width, in pixels, of search box for block matching
                          algorithm. default 3.
    :parameter int height: height, in pixels, of search box for block matching
                          algorithm. default 3.

    :param float  alpha: The background learning factor, its value should
                       be between 0 and 1. The higher the value, the more quickly
                       your program learns the changes in the background. Therefore,
                       for a static background use a lower value, like 0.001. But if
                       your background has moving trees and stuff, use a higher
                       value, maybe start with 0.01.
                       default: 0.01

    Return
    ------
        :return 2d_array background:
        :return list mean_velocity:
        :return list layers:
    '''
    def __init__(self, func):
        '''Constructor.'''
        self._width = 3
        self._height = 3
        self._first_frame = True
        self._background = None
        self._alpha = 0.01
        self._old_frame = None
        self.func = func


    def __call__(self, *args, **kwargs):
        '''
        Layer decorator.

        Parameters
        ----------

        :parameter int width: width, in pixels, of search box for block matching
                              algorithm. default 3.
        :parameter int height: height, in pixels, of search box for block matching
                              algorithm. default 3.

        :param float  alpha: The background learning factor, its value should
                           be between 0 and 1. The higher the value, the more quickly
                           your program learns the changes in the background. Therefore,
                           for a static background use a lower value, like 0.001. But if
                           your background has moving trees and stuff, use a higher
                           value, maybe start with 0.01.
                           default: 0.01

        Return
        ------
            :return 2d_array background:
            :return list mean_velocity:
            :return list layers:
        '''

        if 'alpha' in kwargs:
            self._alpha = kwargs['alpha']

        if 'width' in kwargs:
            self._width = kwargs['width']

        if 'height' in kwargs:
            self._height = kwargs['height']

        for frame in self.func(*args, **kwargs):

            if self._first_frame is True:
                self._background = BackgroundSubtractor(self._alpha, frame)
                self._old_frame = self._background.foreground(frame)
                self._first_frame = False
            else:
                foreground = self._background.foreground(frame)
                XP, YP, XD, YD = block_matching(self._old_frame,
                                                foreground,
                                                self._width,
                                                self._height)

                U, V, object_tops, meand = clustering(XD, YD, XP, YP)

                self._old_frame = foreground.copy()
                


            yield self._background.background
