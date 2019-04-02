"""
Background subtraction algorithm.

The algorithm is based in the work of [YiFan2010]_.

Example
--------
>>> import cv2
>>> from blockmatching import *
>>> from numpy import  hstack
>>> cap = cv2.VideoCapture('./videos/car.mp4')
>>> started = False
>>> old_frame = None
>>> background = Nonea
>>> # Learning rate
>>> alpha = 0.01
>>>
>>> while cap.isOpened():
>>>    ret, frame = cap.read()
>>>    if ret == True:
>>>        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
>>>        if started is False:
>>>            background = BackgroundSubtractor(alpha, frame)
>>>            foreground = background.foreground(frame)
>>>            started = True
>>>        else:
>>>            foreground = background.foreground(frame)
>>>            img = hstac(background.background, foreground)
>>>            cv2.imshow('Background x Foreground', img)
>>>            if cv2.waitKey(25) & 0xFF == ord('q'):
>>>                break
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
.. [YiFan2010] YI, Zheng; LIANGZHONG, Fan. Moving object detection based on running
average background and temporal difference. In: 2010 IEEE International
Conference on Intelligent Systems and Knowledge Engineering.
IEEE, 2010. p. 270-272.

"""
import cv2
from numpy import uint8, logical_and
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt2d


class BackgroundSubtractor:
    r'''
    Background estimation and subtraction.

    :param float  alpha: The background learning factor, its value should
                       be between 0 and 1. The higher the value, the more quickly
                       your program learns the changes in the background. Therefore,
                       for a static background use a lower value, like 0.001. But if
                       your background has moving trees and stuff, use a higher
                       value, maybe start with 0.01.

    :param 2d_array firstFrame: This is the first frame.
    '''

    def __init__(self, alpha, firstFrame, threashold=10, sgm=3):
        r'''

        :param float  alpha: The background learning factor, its value should
                           be between 0 and 1. The higher the value, the more quickly
                           your program learns the changes in the background. Therefore,
                           for a static background use a lower value, like 0.001. But if
                           your background has moving trees and stuff, use a higher
                           value, maybe start with 0.01.

            :param 2d array firstFrame: This is the first frame.
        '''
        self.alpha = alpha
        self._sgm = sgm
        self._background = gaussian_filter(firstFrame, sigma=self._sgm)
        self._oldf = self._background.copy()
        self._threashold = threashold

    @property
    def background(self):
        r'''
        Learned background.

        :return 2d_array background: 2d array representing the learned background
        '''
        return gaussian_filter(self._background.astype(uint8), sigma=self._sgm)

    def foreground(self, frame):
        r'''
        Get Foreground.
        Apply the background averaging formula:

        .. math::

            bkg_\{i\} = frame_\{i\} * \alpha + bkg_\{i-1\} * (1 - \alpha)

        where :math:`bkg_\{i\}` is the current background, :math:`\alpha` is
        the learning rate and :math:`frame_\{i\}` is the current frame.

        :param 2d_array frame: Current frame.

        :return 2d_array foreground: the estimated foreground.
        '''
        diff = cv2.subtract(self._oldf, frame)
        diff = gaussian_filter(diff, sigma=self._sgm)
        diff[diff < self._threashold] = 0
        diff[diff > self._threashold] = 255
        self._oldf = frame
        fclean = gaussian_filter(frame, sigma=self._sgm)

        self._background = cv2.addWeighted(fclean, self.alpha,
                                             self._background,
                                             (1.0 - self.alpha),
                                             0.0)

        result = cv2.absdiff(self._background.astype(uint8), frame)
        result[result < self._threashold] = 0
        result[result > self._threashold] = 255
        tmp = result.copy()
        tmp[~logical_and(diff, result)] = 0

        return medfilt2d(tmp, kernel_size=self._sgm)
