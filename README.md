# StemMove
Tools to analyse tree stem movement from 3D point cloud series.

This repository provides three utilities:
- `pc2line.py`, to extract the spine of a point cloud. The spine connects 2 points.
- `pc2segmented.py`, to extract the spine along a series of user-defined points.
- `show_spine.py`, to visualize a .LAS file and the fitted spine.


`pc2line.py` extracts the shortest path throught a given point cloud linking 2
end points specified by the user. Raw and smoothed version of the path lines
are written to disk (as `numpy` arrays).
This is an implementation of Iddo Haniel's 'spine-finding' algorithm as provided at
<https://stackoverflow.com/questions/64911820/fit-curve-spline-to-3d-point-cloud>.

`pc2segmented.py` does the same thing as `pc2line.py` but on a succession of end points. This was developped to handle point cloud hulls with marked curvature and constrain spine to a preferred path.

`show_spine.py` can be used to explore point cloud data and display the spine
lines generated by the script above.

![Spine-finding algorithm](spine_finding_raw_smoothed.png)

Spatial coordinates of the queried end points in the above example are:

    - p0 = (-0.046     0.183  0.003)
    - p1 = (0.015   0.071   0.3)
   
## Usage info

CLI:
```
python -O pc2line.py Cloud.las -0.046 0.183 0.003 0.015 0.071 0.3

python -O pc2segmented.py Cloud.las endpoints.csv
```
The provided PCD (3,000 points) is processed in 2.1s. A dataset with 15 times more points (~45,000) is processed in 19.2s. For 88,000 pts, the script runs in 32 s. Behaviour largely appears O(n). Max. tet size (`r_thresh`) has a notable impact on processing time (because it reduces total number of tets to process). Using PCD with the highest number of points possible allows very small tetrahedra size to be specified and therefore to avoid lumping needles and branches with the stem (see below re. cutting corners).

A poor choice of end coordinates will break down the algorithm. Increase the rthreshold value to avoid the issue mentioned above (likely caused by disconnected cells at low threshold values).

The smoothing paramter is extremely important to control the spline shape. Values are typically very low (e.g. 1.0e-6). The alpha shape parameter of the Voronoi mesher is also relatively important. Too large and the shortest path cut corners and do not track the spine. Too small and the mesh is split into disconnected sub-meshes (you'll get `networkx.exception.NetworkXNoPath: No path between [...]`. It is best to adjust those for each input PCD. They can only be accessed from the code in the current version (2022-10-05).

   
