#!/usr/bin/env python3.6
# -*- Coding: UTF-8 -*-
'''
Using data from block matching algorithm to separeting moving regions in
image using graph theory.

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

   Bibliography:
   BRAD, Remus; LETIA, Ioan Alfred. Cloud motion detection from infrared
   satellite images. In: Second International Conference on Image and Graphics.
   International Society for Optics and Photonics, 2002. p. 408-413.
'''

import networkx as nx
from numpy import abs, array, sqrt,  where, zeros_like, floor, median
import matplotlib.pyplot as plt

def _mout_edges(nodes):
    """Find edges using vertices representing xy position vertices."""
    n = nodes.shape[0]
    edges = []
    for i in range(0, n - 1):
        for j in range(i, n):
            if abs(nodes[i, 0] - nodes[j, 0]) > 1:
                break
            elif abs(nodes[i, 0] - nodes[j, 0]) == 1 and \
                 abs(nodes[i, 1] - nodes[j, 1]) == 0:
                 edges.append([i, j])
            elif abs(nodes[i, 1] - nodes[j, 1]) == 1:
                edges.append([i, j])
    return edges


def clustering(x0, y0, x1, y1, smooth=15):
    """
    Estimating displacement of objects using optical flow.

    Based on connected components in an undirected graph algorithm.

    Generate a mask with arrows representing the vector moviment.
    Velocities are smoothed from median following the sugestion from the work
    of Bran and Letia, 2002 (doi:10.1117/12.477174).
    Parameters
    ----------
    :parameter 2d_array x0: 2d Array - Grid with x initial position of vector
    :parameter 2d_array y0: 2d Array - Grid with y initial position of vector
    :parameter 2d_array x1: 2d Array - Grid with x final position of vector
    :parameter 2d_array y1: 2d Array - Grid with y final position of vector

    Return
    ------
    :return 2d_array dsx: 2d Array - Grid with x displacement.
    :return 2d_arraydsy: 2d Array - Grid with y displacement.
    :return list object_tops: list of lists with x, y for each graph.
    :return list mean_displacement: list with mean displacement of each graph.
    """
    sh = x0.shape
    ds = sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2.0)
    dsx = zeros_like(x1)
    dsy = zeros_like(y1)
    no_zeros = where(abs(ds) != 0.)
    nodes = list(zip(no_zeros[0], no_zeros[1]))
    nnodes = len(nodes)
    nodes = array(nodes)
    edges = _mout_edges(nodes)
    graph = nx.Graph()
    nodes_names = list(range(nnodes - 1))
    graph.add_nodes_from(nodes_names)
    graph.add_edges_from(edges)
    object_tops = []
    mean_displacement = []

    for subgraph in nx.connected_components(graph):

        ij = array(nodes[list(subgraph)])
        ij = (ij[:,0], ij[:, 1])
        n = ij[0].shape[0]

        mdsx = floor(median(x0[ij] - x1[ij]))
        mdsy = floor(median(y0[ij] - y1[ij]))

        nnodes = ij[0].size

        if nnodes <= smooth:
            dsx[ij] = mdsx
            dsy[ij] = mdsy
        else:
            for smo_i in range(0, nnodes - smooth, smooth):
                subij = (ij[0][smo_i:smo_i + smooth], ij[1][smo_i:smo_i + smooth])
                smdsx = floor(median(x0[subij] - x1[subij]))
                smdsy = floor(median(y0[subij] - y1[subij]))

                dsx[subij] = smdsx
                dsy[subij] = smdsy

        object_tops.append(list(zip(x1[ij], y1[ij])))
        mean_displacement.append([mdsx, mdsy])

    return dsx, dsy, object_tops, mean_displacement
