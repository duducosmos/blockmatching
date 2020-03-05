#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
'''
Using opencv to create the vector field using data from block matching
algorithm.

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
>>>            maskvector = vectormask(frame, XD, YD,
>>>                                    (XD + U).astype(int),
>>>                                    (YD + V).astype(int))
>>>
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
'''

import cv2
from numpy import zeros_like

def vectormask(image, x0, y0, x1, y1, color=(255, 255, 255), width=1):
    """
    Vector field image mask.

    Generate a mask with arrows representing the vector moviment.
    Parameters
    ----------

    :parameter 2d_array image: 2d Array
    :parameter 2d_array x0: 2d Array - Grid with x initial position of vector
    :parameter 2d_array y0: 2d Array - Grid with y initial position of vector
    :parameter 2d_array x1: 2d Array - Grid with x final position of vector
    :parameter 2d_array y1: 2d Array - Grid with y final position of vector
    :parameter tuple color: - tuple with rgb color
    :parameter int width: - integer - line width

    Return
    ------

    :return 2d_array: array like input image.
    """

    sh = x0.shape

    mask = zeros_like(image)
    for i in range(1, sh[0]):
        for j in range(1, sh[1]):
            mask = cv2.arrowedLine(mask,
                                   (y0[i,j], x0[i,j]),
                                   (y1[i,j], x1[i,j]),
                                    color, width
                                    )
    return mask
