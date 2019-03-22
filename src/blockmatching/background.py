"""
Background subtraction algorithm.

The algorithm is based in the work of :cite:`yi2010moving`.

.. bibliography:: refs.bib
    :style: plain

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

            bkg_{i} = frame_{i} * \alpha + bkg_{i-1} * (1 - \alpha)

        where :math:`bkg_{i}` is the current background, :math:`\alpha` is
        the learning rate and :math:`frame_{i}` is the current frame.

        :param 2d_array frame: Current frame.

        :return 2d_array foreground: the estimated foreground.
        '''
        diff = cv2.subtract(self._oldf, frame)
        diff = gaussian_filter(diff, sigma=self._sgm)
        diff[diff < self._threashold] = 0
        diff[diff > self._threashold] = 255
        self._oldf = frame
        fclean = gaussian_filter(frame, sigma=self._sgm)

        self._background = cv2.add(fclean * self.alpha,
                                   self._background * (1.0 - self.alpha))

        result = cv2.absdiff(self._background.astype(uint8), frame)
        result[result < self._threashold] = 0
        result[result > self._threashold] = 255
        tmp = result.copy()
        tmp[~logical_and(diff, result)] = 0

        return medfilt2d(tmp, kernel_size=self._sgm)
