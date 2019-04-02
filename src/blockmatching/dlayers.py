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
    :parameter int sigma: int - default 7. Used to create a smoothed mask to separete
                     moving areas.

Return
------
    :return 2d_array background: Background in video
    :return 2d_array foreground: foreground in video
    :return 2d_array vectormask: vector field image
    :return list mean_velocity: mean velocity of connected objects
    :return list layers: connected objects separated in layers.

Example
-------

>>> import cv2
>>> from blockmatching import *
>>>
>>> @dlayers(alpha=0.01, width=3, height=3)
>>> def background(videofile):
>>>     cap = cv2.VideoCapture(videofile)
>>>     while cap.isOpened():
>>>         ret, frame = cap.read()
>>>         if ret == True:
>>>             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
>>>             yield frame
>>>
>>> for bg, fg, mask, meand, layers in background("./videos/car.mp4"):
>>>     fg = cv2.resize(bg, (0,0), fx=0.5, fy=0.5)
>>>
>>>     if layers:
>>>         cv2.imshow("Layer 0", layers[0])
>>>
>>>     cv2.imshow('Background', fg)
>>>     if cv2.waitKey(25) & 0xFF == ord('q'):
>>>         break
"""


from .blockmatching import block_matching
from .background import BackgroundSubtractor
from .clustering import clustering
from .motionlayers import layers
from .vectormask import vectormask
import cv2


def dlayers(alpha=0.01, width=9, height=9, sigma=7):
    '''
    Layer decorator.

    Parameters
    ----------

    :parameter int width: width, in pixels, of search box for block matching
                          algorithm. default 9.
    :parameter int height: height, in pixels, of search box for block matching
                          algorithm. default 9.

    :param float  alpha: The background learning factor, its value should
                       be between 0 and 1. The higher the value, the more quickly
                       your program learns the changes in the background. Therefore,
                       for a static background use a lower value, like 0.001. But if
                       your background has moving trees and stuff, use a higher
                       value, maybe start with 0.01.
                       default: 0.01

    :parameter int sigma: int - default 7. Used to create a smoothed mask to separete
                     moving areas.

    Return
    ------
        :return 2d_array background: Background in video
        :return 2d_array foreground: foreground in video
        :return 2d_array vectormask: vector field image
        :return list mean_velocity: mean velocity of connected objects
        :return list layers: connected objects separated in layers.

    Example
    -------

    >>> import cv2
    >>>
    >>>
    >>> @layers(alpha=0.01, width=3, height=3)
    >>> def background(videofile):
    >>>     cap = cv2.VideoCapture(videofile)
    >>>     while cap.isOpened():
    >>>         ret, frame = cap.read()
    >>>         if ret == True:
    >>>             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    >>>             yield frame
    >>>
    >>> for bg, fg, mask, meand, layers in background("./videos/car.mp4"):
    >>>     fg = cv2.resize(bg, (0,0), fx=0.5, fy=0.5)
    >>>
    >>>     if layers:
    >>>         cv2.imshow("Layer 0", layers[0])
    >>>
    >>>     cv2.imshow('Background', fg)
    >>>     if cv2.waitKey(25) & 0xFF == ord('q'):
    >>>         break
    '''
    def wrap(func):
        def wrapped_func(*args, **kwargs):

            first_frame = True
            background = None
            foreground = None
            maskvector = None
            old_frame = None
            meand = []
            lyrs = []

            for frame in func(*args, **kwargs):
                if first_frame is True:

                    background = BackgroundSubtractor(alpha, frame)
                    foreground = background.foreground(frame)
                    old_frame = foreground.copy()
                    first_frame = False

                else:

                    foreground = background.foreground(frame)

                    XP, YP, XD, YD = block_matching(old_frame,
                                                    foreground,
                                                    width,
                                                    height)

                    U, V, object_tops, meand = clustering(XD, YD, XP, YP)

                    lyrs = layers(frame,
                                  object_tops,
                                  width,
                                  height,
                                  sigma=sigma)

                    old_frame = foreground.copy()
                    maskvector = vectormask(foreground,
                                            XD,
                                            YD,
                                            (XD + U).astype(int),
                                            (YD + V).astype(int))

                yield background.background, foreground, maskvector, meand, lyrs
        return wrapped_func
    return wrap
