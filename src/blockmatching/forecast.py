#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
"""
Forecast decorator.

Decorated function must be a generator that return frame images.

The function will be return the current frame and the forecasting
image of \'connected\' moving objects in the video.

Parameters
----------
    :parameter int minutes: forecasting time in minutes.

    :parameter int dt: delta time - time range from frame to frame.

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
    :return 2d_array background: Current frame;
    :return 2d_array foreground: Forecasting frame;

Example
-------

>>> import cv2
>>> from blockmatching import *
>>>
>>> @forecasting(minutes=30, alpha=0.01, width=3, height=3)
>>> def background(videofile):
>>>     cap = cv2.VideoCapture(videofile)
>>>     while cap.isOpened():
>>>         ret, frame = cap.read()
>>>         if ret == True:
>>>             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
>>>             yield frame
>>>
>>> for current, forecasting in background("./videos/car.mp4"):
>>>
>>>     if forecasting is not None:
>>>         cv2.imshow("Forecasting, forecasting)
>>>
>>>     if cv2.waitKey(25) & 0xFF == ord('q'):
>>>         break
"""
from numpy import zeros_like, pad, ones, float32

from .blockmatching import block_matching
from .background import BackgroundSubtractor
from .clustering import clustering
from .motionlayers import layers
import cv2


def forecasting(seconds, dt, alpha=0.01, width=9, height=9, sigma=7):
    """
    Forecasting using block matching algorithm.

    Parameters
    ----------

        :parameter int seconds: forecasting time in seconds.

        :parameter int dt: delta time - time range in seconds from frame to frame.

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
        :return 2d_array background: Current frame;
        :return 2d_array foreground: Forecasting frame;

    Example
    -------

    >>> import cv2
    >>> from blockmatching import *
    >>>
    >>> @forecasting(seconds=1800, dt= 1800, alpha=0.01, width=3, height=3)
    >>> def background(videofile):
    >>>     cap = cv2.VideoCapture(videofile)
    >>>     while cap.isOpened():
    >>>         ret, frame = cap.read()
    >>>         if ret == True:
    >>>             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    >>>             yield frame
    >>>
    >>> for frame, forecasting in background("./videos/car.mp4"):
    >>>
    >>>     if forecasting is not None:
    >>>         cv2.imshow("Forecasting, forecasting)
    >>>
    >>>     if cv2.waitKey(25) & 0xFF == ord('q'):
    >>>         break
    """
    def wrap(func):
        def wrapped_func(*args, **kwargs):

            first_frame = True
            forecast = None
            background = None
            foreground = None
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

                    lars = layers(frame,
                                  object_tops,
                                  width,
                                  height,
                                  sigma=sigma)

                    old_frame = foreground.copy()

                    r = zeros_like(frame)
                    for i in range(len(meand)):
                        c = seconds / dt
                        dsy, dsx = int(c * meand[i][0]) , int(c * meand[i][1])

                        fill_width = (0, abs(dsx)) if dsx < 0 \
                                     else (abs(dsx), 0)
                        fill_height = (0, abs(dsy)) if dsy < 0 \
                                     else (abs(dsy), 0)

                        tmp = pad(lars[i],
                                  (fill_height, fill_width),
                                  mode='constant')

                        if dsy == 0 and dsx < 0:
                            tmp = tmp[:, abs(dsx):]
                        elif dsy == 0 and dsx > 0:
                            tmp = tmp[:, :-abs(dsx)]
                        elif dsy > 0 and dsx == 0:
                            tmp = tmp[:-abs(dsy), :]
                        elif dsy < 0 and dsx == 0:
                            tmp = tmp[abs(dsy):, :]
                        elif dsy < 0 and dsx < 0:
                            tmp = tmp[abs(dsy):, abs(dsx):]
                        elif dsy < 0 and dsx > 0:
                            tmp = tmp[abs(dsy):, :-abs(dsx)]
                        elif dsy > 0 and dsx < 0:
                            tmp = tmp[:-abs(dsy), abs(dsx):]
                        elif dsy > 0 and dsx > 0:
                            tmp = tmp[:-abs(dsy), :-abs(dsx)]

                        r = r + tmp

                    kernel = ones((sigma, sigma), float32) / sigma ** 2
                    forecast = cv2.filter2D(r, -1, kernel)
                    #forecast = cv2.add(background.background, forecast)

                yield frame, forecast, foreground, background.background
        return wrapped_func
    return wrap
