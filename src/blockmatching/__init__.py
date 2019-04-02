#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
"""
Block Matching Algorithm to estimate optical flux and detection of moving
objects.


According to [cuevs2013]_ in a block matching (BM) approach:

    '''...image frames in a video sequence are divided into blocks. For each
    block in the current frame, the best matching block is identified inside a
    region of the previous frame, aiming to minimize the sum of absolute
    differences...'''

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
"""

from .blockmatching import *
from .clustering import *
from .vectormask import *
from .background import *
from .motionlayers import *
from .dlayers import *
from .savevideo import *
from .forecast import *
