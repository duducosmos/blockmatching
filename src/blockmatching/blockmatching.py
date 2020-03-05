#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
"""
Block Matching Algorithm.
------------------------
According to [cuevs2013]_ in a block matching (BM) approach:

    '...image frames in a video sequence are divided into blocks. For each
    block in the current frame, the best matching block is identified inside a
    region of the previous frame, aiming to minimize the sum of absolute
    differences...'

From the work of  [perez2010]_:

    '...pixel-specific motion vectors are determined by calculating the RMSE of
     the difference between two consecutive Kt*grids surrounding the considered
     pixel when the second grid is advected in the direction of a motion vector.
     The selected motion vector corresponds to the lowest RMSE. This process is
     repeated for each image pixel, and each pixel is assigned an individual
     motion vector. Future images are obtained by displacing the current image
     pixels in the direction of their motion vector. Future images are
     subsequently smoothed by averaging each pixel with its 8 surrounding
     neighbors...'

For example, considering a image, in  :math:`t_0 + k dt`, with 9x9 pixels and a
block grid with 3x3 pixels. The image bellow  it is assumed that the central
pixel C is surrounding by pixels A.

::

    * * * * * * * * *
    * * * * * * * * *
    * * * * * * * * *
    * * * A A A * * *
    * * * A C A * * *
    * * * A A A * * *
    * * * * * * * * *
    * * * * * * * * *
    * * * * * * * * *


Now, for a image in time :math:`t_0 + (k+1)dt`, the value of block with the pixel C,
in the image in :math:`t_0 +kdt`, is compared with values of piexls in the 9x9 window
of the image in :math:`t_0+(k+1)dt`.

The most probable direction of the moviment of the pixel C, at
:math:`t_0 + (k+1)dt`, is given by the position of the corresponding block with
the lowest square mean error -SME (subtraction of the 3x3 subgrid)
(e.g. [khawase17]_).

In the following example, the 3x3 block was in the initial position i=4, j=4.
The new initial subblock with lowest  is in i=7, j=7.

Initial position of 3x3 block in :math:`t_0 + kdt`:

::

* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * A A A * * *
* * * A C A * * *
* * * A A A * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *

The new position of 3x3 block in :math:`t_0 + (i+1)dt`:

::

* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * A A A
* * * * * * A C A
* * * * * * A A A

The size of search window depend of the expected velocity of the block. For
slow moviment, a window with size of 3  times can be considered.

:Example:

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

References
----------
.. [cuevs2013] CUEVAS, Erik et al. Block matching algorithm for motion
estimation based on Artificial Bee Colony (ABC).
Applied Soft Computing, v. 13, n. 6, p. 3047-3059, 2013.

.. [khawase17] KHAWASE, Sonam T. et al. An Overview of Block Matching
Algorithms for Motion Vector Estimation. In: Proceedings of the Second
International Conference on Research in Intelligent and Computing in
Engineering, str. 2017. p. 217-222.

.. [perez2010] Perez, R. et al. Validation of short and medium term
operational solar radiation forecasts in the US.  Solar Energy,
84. 12. 2161-2172. 2010.
"""

from numba import jitclass
from numba import njit, prange, jit, float64, int64, uint8

from numpy import zeros, sqrt, array
import cv2


TYPESSME = "float64[:](float64[:,:], float64[:,:], int64, int64)"

TYPESME = "float64[:](float64[:,:], float64[:,:], int64, int64, int64, int64,"
TYPESME += "int64, int64, int64, int64, int64, int64)"

TYPEBBMATCHING = "int64[:,:](float64[:,:], float64[:,:], int64, int64, int64,"
TYPEBBMATCHING += "int64, int64, int64 ,int64, int64,)"

TYPEBMATCHING = "int64[:,:], int64[:,:], int64[:,:], int64[:,:](float64[:,:],"
TYPEBMATCHING += "float64[:,:], int64, int64)"


class BlockError(Exception):
    def __init__(self, value):
        value = value

    def __str__(self):
        return repr(value)


class ImageSizeError(Exception):
    def __init__(self, value):
        value = value

    def __str__(self):
        return repr(value)


@njit(TYPESSME, nogil=True)
def _ssme(window, block, height, width):
    r'''
    Compare the block in the window.
    input:
        window - 2d array representing the window
        blcok - 2d array representing the block to match
        height - int64 - height of block
        width - int64 - height of block
    '''
    out = zeros(3)
    minval = 1e99
    if block.shape[0] != height or block.shape[1] != width:
        out[0] = height // 2
        out[1] = width // 2
        return out

    for ki in prange(height):
        for kj in prange(width):
            tmp = (window[ki: ki + height, kj: kj + width] - block).flatten()
            diff = 0
            for val in list(tmp):
                if val < 0:
                    val = 0
                if val > 255:
                    val = 255
                diff += val
            diff = diff / (height * width)

            if ki == height // 2 and kj == width // 2:

                # To Priorize the central region of window.
                # Neighbors can have same value and false moviment is detected.
                # To avoid, a diferent approach for center of window.
                if diff <= minval:
                    minval = diff
                    out[0] = ki
                    out[1] = kj
                    out[2] = minval

            if(diff < minval):
                minval = diff
                out[0] = ki
                out[1] = kj
                out[2] = minval

    return out


@njit(TYPESME, nogil=True)
def _sme(img1, img0, wblock, hblock, wwind, hwind,
        width, height, i0, i1, j0, j1):
    '''
    Create window and blocks to matching searching.
    input:
        img0 - 2d float64 array - image in time t + idt
        img1 - 2d float64 array - image in time t + (i +1 )dt
        wblock - int64 - Number of blocks in width
        hblock - int64 - Number of blocks in height
        wwind - int64 - Window width
        hwind - int64 - Window height
        width - int64 - block width in pixels
        height - int64 - block height in pixels
        i0 - int64 - initial line of window
        i1 - int64 - final line of window
        j0 - int64 - initial column of window
        j1 - int64 - final column of window
    Return:
        Integer array with four elements:
            inital line, initial column,
            final line, final column.
    '''
    out = zeros(4)

    window = img1[i0: i1, j0: j1]

    jb0 = j0 + width // 2
    jb1 = j0 + 3 * width // 2

    ib0 = i0 + height // 2
    ib1 = i0 + 3 * height // 2
    block = img0[ib0: ib1, jb0: jb1]

    tmp = _ssme(window, block, height, width)
    out[0] = ib0
    out[1] = jb0
    out[2] = i0 + tmp[0]
    out[3] = j0 + tmp[1]
    return out


@njit(TYPEBBMATCHING, parallel=True, nogil=True)
def _block_matching(img0, img1, width, height, wblock, hblock,
                    wwind, hwind, wwindt, hwindt):
    """
    Divide image in windows to run in parallel mode.
    input:
        img0 - 2d float64 array - image in time t + idt
        img1 - 2d float64 array - image in time t + (i +1 )dt
        width - int64 - block width in pixels
        height - int64 - block height in pixels
        wblock - int64 - Number of blocks in width
        hblock - int64 - Number of blocks in height
        wwind - int64 - Window width
        hwind - int64 - Window height
        wwindt - int64 - Total number of windows in width
        hwindt - int64 - Total number of windows in height
    Return:
        Array with nxm lines and 4 collumns.
        Each collumn:
            0 - Initial x
            1 - Initial y
            2 - Final matching x
            3 - Final matching y
    """
    m = hblock * wblock
    out = zeros((m, 4), dtype=int64)
    for i in prange(hblock - 1):
        for j in prange(wblock - 1):
            i0 = i * (height)
            i1 = (i + 1) * (hwind)

            j0 = j * (width)
            j1 = (j + 1) * (wwind)

            x = _sme(img1, img0, wblock, hblock, wwind, hwind,
                     width, height, i0, i1, j0, j1)

            k0 = i * wblock + j
            out[k0, :] = x
    return out


def block_matching(img0, img1, width, height):
    """
    Block matching algorithm.
    -------------------------
    Motion estimation from two sequential images.

    :Example:

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
    >>>            old_frame = frame
    >>>    else:
    >>>         break
    >>>
    >>> cap.release()

    Parameters
    ----------
        :parameter 2d_array img0: 2d array - Image in time t0 + kdt
        :parameter 2d_array img1: 2d array - Image in time t0 + (k+1)dt
        :parameter int64 width: int64 - matching block width
        :parameter int64 height: int64 - matching block height

    Return:
    ------
        :return 2d_array XI: - 2d int64 array - Grid with Initial x
        :return 2d_array YI: - 2d int64 array - Grid with Initial y
        :return 2d_array XF: - 2d int64 array - Final matching Grid with x
        :return 2d_array YF: - 2d int64 array - Final matching Grid with y
    """

    if img0.shape[0] != img1.shape[0]:
        raise ImageSizeError("The images have diferent number of lines")

    if img0.shape[1] != img1.shape[1]:
        raise ImageSizeError("The images have diferent number of columns")

    lins, cols = img0.shape

    # Number of blocks in width
    wblock = cols // width

    # Number of blocks in height
    hblock = lins // height

    if wblock < 2:
        raise BlockError("block larger than image.")

    if hblock < 2:
        raise BlockError("block heigher than image.")

    # Window width
    wwind = 2 * width

    # Window height
    hwind = 2 * height

    # Total windows in width
    wwindt = cols // wwind

    # Total windows in height
    hwindt = lins // hwind
    out = _block_matching(img0.astype(float), img1.astype(float), width,
                          height, wblock, hblock, wwind, hwind, wwindt, hwindt
                          )

    return out[:, 0].reshape(hblock, wblock), out[:, 1].reshape(hblock, wblock),\
           out[:, 2].reshape(hblock, wblock), out[:, 3].reshape(hblock, wblock),
