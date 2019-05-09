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
'''

import networkx as nx
from numpy import abs, array, sqrt,  where, zeros_like, floor
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


def clustering(x0, y0, x1, y1):
    """
    Estimating displacement of objects using optical flow.

    Based on connected components in an undirected graph algorithm.

    Generate a mask with arrows representing the vector moviment.
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

        '''
        graphTMP = nx.Graph()
        graphTMP.add_edges_from(nodes[list(subgraph)])
        nx.draw(graphTMP)
        plt.show()
        print(nx.subgraph_centrality(graphTMP))
        '''
        
        ij = array(nodes[list(subgraph)])
        ij = (ij[:,0], ij[:, 1])
        n = ij[0].shape[0]

        mdsx = floor(sqrt((x0[ij] - x1[ij]) ** 2.0).sum() / n)
        mdsy = floor(sqrt((y0[ij] - y1[ij]) ** 2.0).sum() / n)

        hati = ((x0[ij] - x1[ij])).sum()
        hati = hati / abs(hati) if abs(hati) != 0 else 0
        hatj = ((y0[ij] - y1[ij])).sum()
        hatj = hatj / abs(hatj) if abs(hatj) != 0 else 0
        mdsx = int(hati * mdsx)
        mdsy = int(hatj * mdsy)
        dsx[ij] = mdsx
        dsy[ij] = mdsy
        #dsx[ij] = (x1[ij] - x0[ij])
        #dsy[ij] = (y1[ij] - y0[ij])
        object_tops.append(list(zip(x1[ij], y1[ij])))
        mean_displacement.append([mdsx, mdsy])

    return dsx, dsy, object_tops, mean_displacement
