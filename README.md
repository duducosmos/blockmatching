# Block Matching Algorithm


## Block Matching

According to CUEVAS et al. (2013) in a block matching (BM) approach:

    "...image frames in a video sequence are divided into blocks. For each
    block in the current frame, the best matching block is identified inside a
    region of the previous frame, aiming to minimize the sum of absolute
    differences..."

From the work of  Perez et al. (2010):

    "...pixel-specific motion vectors are determined by calculating the RMSE of
     the difference between two consecutive Kt*grids surrounding the considered
     pixel when the second grid is advected in the direction of a motion vector.
     The selected motion vector corresponds to the lowest RMSE. This process is
     repeated for each image pixel, and each pixel is assigned an individual
     motion vector. Future images are obtained by displacing the current image
     pixels in the direction of their motion vector. Future images are
     subsequently smoothed by averaging each pixel with its 8 surrounding
     neighbors..."

For example, considering a image, in  t0 + k dt, with 9x9 pixels and a block
grid with 3x3 pixels. The image bellow  it is assumed that the central pixel C
is surrounding by pixels A.

```
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * A A A * * *
* * * A C A * * *
* * * A A A * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
```

Now, for a image in time t0 + (k+1)dt, the value of block with the pixel C,
in the image in t0 +kdt, is compared with values of piexls in the 9x9 window
of the image in t0+(k+1)dt.

The most probable direction of the moviment of the pixel C, at t0 + (k+1)dt,
is given by the position of the corresponding block with the lowest
square mean error -SME (subtraction of the 3x3 subgrid) (e.g. KHAWASE et al. (2017)).

In the following example, the 3x3 block was in the initial position i=4, j=4.
The new initial subblock with lowest  is in i=7, j=7.

Initial position of 3x3 block in t0 + kdt:

```
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * A A A * * *
* * * A C A * * *
* * * A A A * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
```
The new position of 3x3 block in t0 + (i+1)dt:

```
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * * * *
* * * * * * A A A
* * * * * * A C A
* * * * * * A A A
```

The size of search window depend of the expected velocity of the block. For
slow moviment, a window with size of 3  times can be considered.


## Background subtraction algorithm.

The background and foreground subtraction algorithm is based on the work of
Yi and Fan (2010):

  "...based on running average background modeling and temporal difference
   method.Firstly, we utilize the running average method to dynamically
   updating the background image. Through using background subtraction, we get
   a foreground image. Secondly, we use temporal difference method to get a
   difference image..."

## License

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

## Bibliography
CUEVAS, Erik et al. Block matching algorithm for motion
estimation based on Artificial Bee Colony (ABC).
Applied Soft Computing, v. 13, n. 6, p. 3047-3059, 2013.

KHAWASE, Sonam T. et al. An Overview of Block Matching
Algorithms for Motion Vector Estimation. In: Proceedings of the Second
International Conference on Research in Intelligent and Computing in
Engineering, str. 2017. p. 217-222.

Perez, R. et al. Validation of short and medium term
operational solar radiation forecasts in the US. Solar Energy,
84. 12. 2161-2172. 2010.

Yi, Zheng, and Fan Liangzhong. Moving object detection
based on running average background and temporal difference. 2010
IEEE International Conference on Intelligent Systems and
Knowledge Engineering. IEEE, 2010.
