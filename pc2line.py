#!/usr/bin/env python
#-*- coding: utf-8 -*-

# DS, 2022-06-07
# Code to extract the stem line from a point cloud.
# For Robin Hartley, NS-TIP.WP4

# Source: https://stackoverflow.com/questions/64911820/fit-curve-spline-to-3d-point-cloud

import sys
import os.path
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import UnivariateSpline
import numpy as np

# For visualising the point cloud and its triangulation
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# import liblas # python -m pip install liblas. Minor version issue on reading
# RH .laz file.
import laspy
# ^~~~ pip install laszip too or no backend error on reading laz .file

import networkx as nx


# --------------------------------------------------------------------------- #
def usg():
    print("Usage:\npython pc2line.py LASFILE.")
    sys.exit()
# --------------------------------------------------------------------------- #


def plot_tri_simple(ax, points, tri=None):
    if tri is not None:
        for tr in tri.simplices:
            pts = points[tr, :]
            ax.plot3D(pts[[0,1],0], pts[[0,1],1], pts[[0,1],2], color='g', lw='0.1')
            ax.plot3D(pts[[0,2],0], pts[[0,2],1], pts[[0,2],2], color='g', lw='0.1')
            ax.plot3D(pts[[0,3],0], pts[[0,3],1], pts[[0,3],2], color='g', lw='0.1')
            ax.plot3D(pts[[1,2],0], pts[[1,2],1], pts[[1,2],2], color='g', lw='0.1')
            ax.plot3D(pts[[1,3],0], pts[[1,3],1], pts[[1,3],2], color='g', lw='0.1')
            ax.plot3D(pts[[2,3],0], pts[[2,3],1], pts[[2,3],2], color='g', lw='0.1')

    ax.scatter(points[:,0], points[:,1], points[:,2], alpha=0.25)


def compute_delaunay_tetra_circumcenters(dt):
    """
    Compute the centers of the circumscribing circle of each tetrahedron in the Delaunay triangulation.
    :param dt: the Delaunay triangulation
    :return: array of xyz points
    """
    simp_pts = dt.points[dt.simplices]
    # (n, 4, 3) array of tetrahedra points where simp_pts[i, j, :] holds the j'th 3D point (of four) of the i'th tetrahedron
    assert simp_pts.shape[1] == 4 and simp_pts.shape[2] == 3

    # finding the circumcenter (x, y, z) of a simplex defined by four points:
    # (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x1)**2 + (y-y1)**2 + (z-z1)**2
    # (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x2)**2 + (y-y2)**2 + (z-z2)**2
    # (x-x0)**2 + (y-y0)**2 + (z-z0)**2 = (x-x3)**2 + (y-y3)**2 + (z-z3)**2
    # becomes three linear equations (squares are canceled):
    # 2(x1-x0)*x + 2(y1-y0)*y + 2(z1-z0)*y = (x1**2 + y1**2 + z1**2) - (x0**2 + y0**2 + z0**2)
    # 2(x2-x0)*x + 2(y2-y0)*y + 2(z2-z0)*y = (x2**2 + y2**2 + z2**2) - (x0**2 + y0**2 + z0**2)
    # 2(x3-x0)*x + 2(y3-y0)*y + 2(z3-z0)*y = (x3**2 + y3**2 + z3**2) - (x0**2 + y0**2 + z0**2)

    # building the 3x3 matrix of the linear equations
    a = 2 * (simp_pts[:, 1, 0] - simp_pts[:, 0, 0])
    b = 2 * (simp_pts[:, 1, 1] - simp_pts[:, 0, 1])
    c = 2 * (simp_pts[:, 1, 2] - simp_pts[:, 0, 2])
    d = 2 * (simp_pts[:, 2, 0] - simp_pts[:, 0, 0])
    e = 2 * (simp_pts[:, 2, 1] - simp_pts[:, 0, 1])
    f = 2 * (simp_pts[:, 2, 2] - simp_pts[:, 0, 2])
    g = 2 * (simp_pts[:, 3, 0] - simp_pts[:, 0, 0])
    h = 2 * (simp_pts[:, 3, 1] - simp_pts[:, 0, 1])
    i = 2 * (simp_pts[:, 3, 2] - simp_pts[:, 0, 2])

    v1 = (simp_pts[:, 1, 0] ** 2 + simp_pts[:, 1, 1] ** 2 + simp_pts[:, 1, 2] ** 2) - (simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2)
    v2 = (simp_pts[:, 2, 0] ** 2 + simp_pts[:, 2, 1] ** 2 + simp_pts[:, 2, 2] ** 2) - (simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2)
    v3 = (simp_pts[:, 3, 0] ** 2 + simp_pts[:, 3, 1] ** 2 + simp_pts[:, 3, 2] ** 2) - (simp_pts[:, 0, 0] ** 2 + simp_pts[:, 0, 1] ** 2 + simp_pts[:, 0, 2] ** 2)

    # solve a 3x3 system by inversion (see https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices)
    A = e*i - f*h
    B = -(d*i - f*g)
    C = d*h - e*g
    D = -(b*i - c*h)
    E = a*i - c*g
    F = -(a*h - b*g)
    G = b*f - c*e
    H = -(a*f - c*d)
    I = a*e - b*d

    det = a*A + b*B + c*C

    # multiplying inv*[v1, v2, v3] to get solution point (x, y, z)
    x = (A*v1 + D*v2 + G*v3) / det
    y = (B*v1 + E*v2 + H*v3) / det
    z = (C*v1 + F*v2 + I*v3) / det

    return (np.vstack((x, y, z))).T


def compute_voronoi_vertices_and_edges(points, r_thresh=np.inf):
    """
    Compute (finite) Voronoi edges and vertices of a set of points.
    :param points: input points.
    :param r_thresh: radius value for filtering out vertices corresponding to
    Delaunay tetrahedrons with large radii of circumscribing sphere (alpha-shape condition).
    :return: array of xyz Voronoi vertex points and an edge list.
    """
    dt = Delaunay(points)
    maya_plot(dt.points, dt.simplices)
    xyz_centers = compute_delaunay_tetra_circumcenters(dt)

    # filtering tetrahedrons that have radius > thresh
    simp_pts_0 = dt.points[dt.simplices[:, 0]]
    radii = np.linalg.norm(xyz_centers - simp_pts_0, axis=1)
    is_in = radii < r_thresh

    # build an edge list from (filtered) tetrahedrons neighbor relations
    edge_lst = []
    for i in range(len(dt.neighbors)):
        if not is_in[i]:
            continue  # i is an outside tetra
        for j in dt.neighbors[i]:
            if j != -1 and is_in[j]:
                edge_lst.append((i, j))

    return xyz_centers, edge_lst


def read_las(filename):
    """Read a laser file and return the 3D point coordinates."""
    with liblas.file.File(filename, mode="r") as fin:
        npts = np.size(fin) # Check that, not sure what I'm getting here
        print("LAS file size:", npts, "Is it number of pts?")
        breakpoint()
        xyz = np.zeros((npts, 3)) # Check that the dims are okay (3xN)
        for i, p in enumerate(fin):
            xyz[i,:] = [p.x, p.y, p.z]

    return xyz


def read_las_v2(filename):
    # The liblas version does not work. This is the laspy version.
    with laspy.open(filename) as fin:
        las = fin.read()
        n = fin.header.point_count
        print(f"Number of points in {filename}: {n}")
        xyz = np.array((las['x'], las['y'], las['z'])).T
        # ^~~~ Dlnay expects <Npts x Ndim>, hence transpose
    return xyz


def spline_3D(xyz, smoothing_factor=0.7):
    """Return a smoothed version of the 3D line coordinates."""
    # Parametric axis
    u = np.arange(xyz.shape[0])
    # ^ Only 'clean' if points are equally-spaced, of which there is 
    # absolutely no guarantee. Use arclength instead, cf. tropism code.

    s = smoothing_factor * xyz.shape[0]

    # UnivariateSpline
    s = 0.7 * len(u)     # smoothing factor
    spx = UnivariateSpline(u, xyz[:,0], s=s)
    spy = UnivariateSpline(u, xyz[:,1], s=s)
    spz = UnivariateSpline(u, xyz[:,2], s=s)

    xyz_new = np.asarray([spx(u), spy(u), spz(u)]).T
    return xyz_new


# ------------------ MAIN -------------------------------------------------- #
def main(las_file):
    # Define extreme points of centroid
    ## Requires a priori knowledge of PC and manual intervention
    start_pt = [16.0, 21.0, 0.0]
    end_pt = [-5.0, 60.0, 40.0]

    # Read laser file data as set of 3 point coordinates
    pts = read_las_v2(las_file)

    # DBG ===
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if __debug__:
        plot_tri_simple(ax, pts)

    # get closest vertex to start and end points
    xyz_centers, edge_lst = compute_voronoi_vertices_and_edges(pts,
                                                               r_thresh=2.0)
    kdt = cKDTree(xyz_centers)
    dist0, idx0 = kdt.query(np.array(start_pt))
    dist1, idx1 = kdt.query(np.array(end_pt))

    # compute shortest weighted path
    edge_lengths = [np.linalg.norm(xyz_centers[e[0], :] - xyz_centers[e[1], :]) for e in edge_lst]
    g = nx.Graph((i, j, {'weight': dist}) for (i, j), dist in zip(edge_lst, edge_lengths))
    path_s = nx.shortest_path(g,source=idx0,target=idx1, weight='weight')
    # ^~~~ shortest_path fails often enough if end points are poorly selected.
    # Define 'poorly'...

    # Get spatial coordinates of path points
    path_xyz = xyz_centers[path_s]

    if __debug__:
        ax.plot(path_xyz[:,0], path_xyz[:,1], path_xyz[:,2], color='red',lw=2,
                label='shortest path')

    # Smooth it
    path_splined = spline_3D(path_xyz)

    if __debug__:
        ax.plot(path_splined[:,0], path_splined[:,1], path_splined[:,2],
                color='chartreuse', lw=2, label='smoothed')
        ax.legend()
        plt.show()

    # Write to disk (as numpy binary format, use load to get in memory again).
    fout = os.path.splitext(las_file)[0] + '_spline.npy'
    print(fout)
    np.save(fout, path_splined)

# -------------------------------------------------------------------------- #


# -------------------------------------------------------------------------- #
if __name__ == "__main__":
    args = sys.argv[:]
    if len(args) != 2:
        usg()

    las_file = args[1]
    if las_file[-4:] not in (".las", ".laz"):
        usg()

    main(las_file)
# -------------------------------------------------------------------------- #

