# coding: utf-8
""" demo on forward 2D """
# Copyright (c) Benyuan Liu. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
from __future__ import division, absolute_import, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.mesh as mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh import quality
from pyeit.mesh.shape import circle

nb_elect = 16

anomalies_list = [
    [
        {"x": 0, "y": 0.5, "d": 0.2, "perm": 10},
        {"x": 0, "y": -0.5, "d": 0.2, "perm": 10},
        {"x": 0.5, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0, "d": 0.2, "perm": 10}
    ],
    [
        # left lung
        {"x": -0.6, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0.2, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": -0.2, "d": 0.2, "perm": 10},
        {"x": -0.4, "y": 0.4, "d": 0.2, "perm": 10},
        {"x": -0.4, "y": -0.4, "d": 0.2, "perm": 10},
        # right lung
        {"x": 0.6, "y": 0, "d": 0.2, "perm": 10},
        {"x": 0.5, "y": 0.2, "d": 0.2, "perm": 10},
        {"x": 0.5, "y": -0.2, "d": 0.2, "perm": 10},
        {"x": 0.4, "y": 0.4, "d": 0.2, "perm": 10},
        {"x": 0.4, "y": -0.4, "d": 0.2, "perm": 10},
    ],
    [
        {"x": 0.5, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0, "d": 0.2, "perm": 10}
    ],
    [
        {"x": 0, "y": 0.5, "d": 0.2, "perm": 10},
        {"x": 0, "y": -0.5, "d": 0.2, "perm": 10},
    ],
    # [
    #     {"x": 0, "y": 0, "d": 0.9, "perm": 10},
    # ]
]


def fem_forward2d(anomaly):
    """ 0. build mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(nb_elect, h0=0.08, fd=circle)

    # extract node, element, alpha
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    x, y = pts[:, 0], pts[:, 1]
    quality.stats(pts, tri)

    # change permittivity
    mesh_obj["alpha"] = np.random.rand(tri.shape[0]) * 200 + 100
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)
    perm = mesh_new["perm"]

    """ 1. FEM forward simulations """
    # setup EIT scan conditions
    ex_dist, step = 1, 1
    ex_mat = eit_scan_lines(nb_elect, ex_dist)
    for ex_line in ex_mat:
        ex_line = ex_line.ravel()
        # Define electrode current sink and current source

        # calculate simulated data using FEM
        fwd = Forward(mesh_obj, el_pos)
        f, _ = fwd.solve(ex_line, perm=perm)
        f = np.real(f)

        """ 2. plot """
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # draw equi-potential lines
        vf = np.linspace(min(f), max(f), 32)
        # Draw contour lines on an unstructured triangular grid.
        ax1.tricontour(x, y, tri, f, vf, cmap=plt.cm.viridis)

        # draw mesh structure
        # Create a pseudocolor plot of an unstructured triangular grid
        ax1.tripcolor(
            x,
            y,
            tri,
            np.real(perm),
            edgecolors="k",
            shading="flat",
            alpha=0.5,
            cmap=plt.cm.Greys,
        )
        # draw electrodes
        ax1.plot(x[el_pos], y[el_pos], "ro")
        for i, e in enumerate(el_pos):
            ax1.text(x[e], y[e], str(i + 1), size=12)
        ax1.set_title("equi-potential lines")
        # clean up
        ax1.set_aspect("equal")
        ax1.set_ylim([-1.2, 1.2])
        ax1.set_xlim([-1.2, 1.2])
        fig.set_size_inches(6, 6)
    plt.show()


if __name__ == "__main__":
    nb_elect = 16
    for anomaly in anomalies_list:
        fem_forward2d(anomaly)
