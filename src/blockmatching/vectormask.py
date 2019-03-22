#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
'''
Using opencv to create the vector field using data from block matching
algorithm.

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
    input:
        image: 2d Array
        x0: 2d Array - Grid with x initial position of vector
        y0: 2d Array - Grid with y initial position of vector
        x1: 2d Array - Grid with x final position of vector
        y1: 2d Array - Grid with y final position of vector
        color - tuple with rgb color
        width - integer - line width
    return:
        array like input image.
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
