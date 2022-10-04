#!/usr/bin/env python
#-*- coding: utf-8 -*-

# DS, 2022-10-04
# Script to convert laser scan data to ascii format (triplets of coords).
# ASCII file can be imported into meshlab for inspection.
# NS-TIP.WP4

import plotly.graph_objects as go
import sys
from pc2line import read_las_v2
import numpy as np

# --------------------------------------------------------------------------- #
def usg():
    print("Usage:\npython show_spine.py LASFILE")
    sys.exit()


def plot(x, y, z, raw=None, smoothed=None):
    # Look at spine dinding results
    if raw is not None and smoothed is not None:
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                           marker={'size':1, 'opacity':0.2}),
                              go.Scatter3d(x=smoothed[:,0], y=smoothed[:,1],
                                           z=smoothed[:,2], mode='lines',
                                           line={'color':'red', 'width':2}),
                              go.Scatter3d(x=raw[:,0], y=raw[:,1], z=raw[:,2],
                                           mode='lines',
                                           line={'color':'green', 'width':2})])
    # Inspect PCD
    else:
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                           marker={'size':1, 'opacity':0.1})])

    fig.show()
# --------------------------------------------------------------------------- #


# ------------------ MAIN -------------------------------------------------- #
def main(las_file, spine_raw, spine_smoothed):
    # Get laser scan data
    pts = read_las_v2(las_file)
#    breakpoint()
    plot(pts[:,0], pts[:,1], pts[:,2], spine_raw, spine_smoothed)


# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = sys.argv[:]
    if len(args) != 2:
        usg()

    las_file = args[1]
    if las_file[-4:] not in (".las", ".laz"):
        print("The input file does not have .las/.laz extension.")
        usg()

    ## get line data.
    try:
        shortest_path = np.load(las_file[:-4] + "_shortest.npy")
    except FileNotFoundError:
        print("Cannot find shortest path data file for PCD.")
        shortest_path = None
        # They are generated automatically by the pc2line.py script

    try:
        centroid = np.load(las_file[:-4] + "_spline.npy")
    except FileNotFoundError:
        print("Cannot find shortest path data file for PCD.")
        centroid = None

    main(las_file, shortest_path, centroid)
# -------------------------------------------------------------------------- #
