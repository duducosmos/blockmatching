#!/usr/bin/env python
"""
Developed by: E. S. Pereira.
e-mail: pereira.somoza@gmail.com
"""

from skvideo.io import FFmpegWriter


class SaveVideo:

    def __init__(self, nfile, rate=2):
        self._nfile = nfile
        self._size = None
        self.out = FFmpegWriter(nfile,
                                inputdict={'-r': str(rate), },
                                outputdict={'-r': str(rate),}
                                )


    def save_frame(self, frame, last=False):
        #stacked_img = np.stack((frame,) * 3, -1)
        #im_color = cv2.applyColorMap(stacked_img, cv2.COLORMAP_OCEAN)
        #im_color = cv2.applyColorMap(stacked_img, cv2.COLORMAP_HOT)

        self.out.writeFrame(frame)

        if last is True:
            self.out.close()
