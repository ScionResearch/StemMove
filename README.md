# StemMove
Tools to analyse tree stem movement from 3D point cloud series.

`pc2line.py` extracts the shortest path throught a given point cloud linking 2
end points specified by the user. Raw and smoothed version of the path lines
are written to disk (as `numpy` arrays).
This is an implementation of the Iddo-Haniel 'spine-finding' algorithm.

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
```
The provided PCD (3,000 points) is processed in 2.1s. A dataset with 15 times more points (~45,000) is processed in 19.2s. Behaviour appears O(n).

The smoothing paramter is extremely important to control the spline shape. Values are typically very low (e.g. 1.0e-6). The alpha shape parameter of the Voronoi mesher is also relatively important. Too large and the shortest path cut corners and do not track the spine. Too small and the mesh is split into disconnected sub-meshes. It is best to adjust those for each input PCD. They can only be accessed from the code in the current version (2022-10-05).
    
