#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
"""
Block Matching Algorithm to estimate optical flux and detection of moving
objects.


According to :cite:`cuevas2013block` in a block matching (BM) approach:

    '''...image frames in a video sequence are divided into blocks. For each
    block in the current frame, the best matching block is identified inside a
    region of the previous frame, aiming to minimize the sum of absolute
    differences...'''

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


.. bibliography:: refs.bib
    :cited:
"""

from .blockmatching import *
from .clustering import *
from .vectormask import *
from .background import *
from .motionlayers import *
