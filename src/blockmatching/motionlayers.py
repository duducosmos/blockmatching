#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
"""
Separete connected moving areas of image in layers.

Example:
--------
>>> import cv2
>>> from blockmatching import *
>>> cap = cv2.VideoCapture('./videos/car.mp4')
>>> started = False
>>> old_frame = None
>>> while cap.isOpened():
>>>    ret, frame = cap.read()
>>>    if ret == True:
>>>        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
>>>        if started is False:
>>>            old_frame = frame
>>>            started = True
>>>        else:
>>>            XP, YP, XD, YD = block_matching(old_frame, frame,
>>>                                            width, height)
>>>            U, V, object_tops, meand = clustering(XD, YD, XP, YP)
>>>            lyrs = layers(frame, object_tops, width, height, sigma=sigma)
>>>            old_frame = frame
>>>    else:
>>>         break
>>>
>>> cap.release()

License
-------
Developed by: E. S. Pereira.
e-mail: pereira.somoza@gmail.com

Copyright [2019] [E. S. Pereira]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
from numpy import zeros_like, ones
from scipy.ndimage.filters import gaussian_filter

def layers(frame, object_tops, width, height, sigma=7):
    '''
    Parameter:
    :parameter 2d_array frame: 2d array representing the current  image frame.
    :parameter list object_tops: list of lists with x, y for each graph from clustering
                     algorithm.
    :parameter int width: int - width used in block matching algorithm (size of block)
    :parameter int height: int - height used in block matching algorithm (size of block)
    :parameter int sigma: int - default 7. Used to create a smoothed mask to separete
                     moving areas.
    Return:
    :return list layers: list of 2d array like frame.
    '''
    layers = []
    for clt in object_tops:
        tmp = zeros_like(frame)
        for i, j in clt:
            tmp[i: i + height, j: j + width] = frame[i: i + height,
                                                     j: j + width]
        dst = gaussian_filter(tmp, sigma=sigma)
        tmp = zeros_like(frame)
        tmp[dst != 0] = frame[dst!=0]
        layers.append(tmp)
    return layers
