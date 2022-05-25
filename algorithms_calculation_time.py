import datetime
import keras.models
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyeit import mesh
from pyeit.eit import bp, jac
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import circle
import tensorflow as tf

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.keras import Sequential


def back_projection(n_el, anomaly):
    """ 0. build mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(n_el=n_el,
                                   fd=circle,
                                   h0=0.1)

    # extract node, element, alpha
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]

    """ 1. problem setup """
    anomaly = anomaly
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

    # draw
    fig, axes = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(6, 4)

    ax = axes[0]
    ax.axis("equal")
    ax.set_title(r"Input $\Delta$ Conductivities")

    delta_perm = np.real(mesh_new["perm"] - mesh_obj["perm"])
    im = ax.tripcolor(
        pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat", cmap=plt.cm.viridis
    )

    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    # adjacent stimulation (el_dist=1), adjacent measures (step=1)
    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(n_el, el_dist)
    # print(ex_mat)

    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])  # default data without perturbation
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])  # data with perturbation
    # print(f1)
    # print(f1.v.shape)

    """
    3. naive inverse solver using back-projection
    """
    eit = bp.BP(mesh_obj, el_pos, ex_mat=ex_mat, step=1, parser="std")
    eit.setup(weight="none")
    ds = eit.solve(f1.v, f0.v, normalize=False)

    # plot
    ax1 = axes[1]
    im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, ds, cmap=plt.cm.viridis)
    ax1.set_title(r"Reconstituted $\Delta$ Conductivities")
    ax1.axis("equal")
    fig.colorbar(im, ax=axes.ravel().tolist())

    # plt.show()


def reconstruction(n_el, anomaly, algo="back-projection", model=None, model2=None, regularization=False, plot=True,
                   raw_measures=False, interpolate=None, autoencoder=False, use_vectorisation=False):
    algo = algo.lower()
    """ 0. construct mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(n_el, h0=0.1, fd=circle)
    # mesh_obj, el_pos = mesh.layer_circle()

    # extract node, element, alpha
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    x, y = pts[:, 0], pts[:, 1]

    """ 1. problem setup """
    anomaly = anomaly
    mesh_obj["alpha"] = np.random.rand(tri.shape[0]) * 200 + 100
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)

    """ 2. FEM simulation """
    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(n_el, el_dist)

    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"], use_vectorization=use_vectorisation)
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"], use_vectorization=use_vectorisation)

    if raw_measures:
        return f1.v

    start_time = datetime.datetime.now()
    ds_n = None
    t_taken = None

    def f():
        nonlocal ds_n, t_taken
        if algo == "back-projection":
            """
            3. naive inverse solver using back-projection
            """
            eit = bp.BP(mesh_obj, el_pos, ex_mat=ex_mat, step=1, parser="std")
            eit.setup(weight="none")
            ds_n = 192.0 * eit.solve(f1.v, f0.v, normalize=False)
            t_taken = datetime.datetime.now() - start_time
            print(f"Back-Projection reconstruction: {t_taken}")
        elif algo == "gauss-newton":
            if regularization:
                """ 3. solve_eit using gaussian-newton (with regularization) """
                # number of stimulation lines/patterns
                eit = jac.JAC(mesh_obj, el_pos, ex_mat, step, perm=1.0, parser="std")
                eit.setup(p=0.25, lamb=1.0, method="lm")
                # lamb = lamb * lamb_decay
                ds_n = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=True)
            else:
                """ 3. JAC solver """
                # Note: if the jac and the real-problem are generated using the same mesh,
                # then, data normalization in solve are not needed.
                # However, when you generate jac from a known mesh, but in real-problem
                # (mostly) the shape and the electrode positions are not exactly the same
                # as in mesh generating the jac, then data must be normalized.
                eit = jac.JAC(
                    mesh_obj,
                    el_pos,
                    ex_mat=ex_mat,
                    step=step,
                    perm=1.0,
                    parser="std",
                )
                eit.setup(p=0.5, lamb=0.01, method="kotre")
                ds = eit.solve(f1.v, f0.v, normalize=True)
                ds_n = sim2pts(pts, tri, np.real(ds))
            t_taken = datetime.datetime.now() - start_time
            print(f"Gauss-Newtonian reconstruction: {t_taken}")

        elif algo == "model":
            if model is not None:
                """4. Model reconstruction"""
                if interpolate is not None:
                    ds_n_model = model.predict(interpolate)[0]
                elif autoencoder:
                    ds_n_model = model.predict(np.array([ds_n]))[0]
                else:
                    ds_n_model = model.predict(np.array([f1.v]))[0]
                t_taken = datetime.datetime.now() - start_time
                print(f"Machine learning: {t_taken}")

    # mem_usage = memory_usage(f)
    mem_usage = [0]
    f()

    return t_taken, max(mem_usage)


def main():
    nb_sim = 50
    str__time = '%M:%S.%f'
    anomalies_list = []
    df_list = []
    df_columns = ["Nb electrodes", "Algorithm", "Average time (s)", "Minimum time (s)", "Maximum time (s)"]
    anomalies = [
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
    ]
    anomalies_list.append(anomalies)
    anomalies = [
        {"x": 0.5, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0, "d": 0.2, "perm": 10}
    ]
    anomalies_list.append(anomalies)
    anomalies = [
        {"x": 0, "y": 0.5, "d": 0.2, "perm": 10},
        {"x": 0, "y": -0.5, "d": 0.2, "perm": 10},
    ]
    anomalies_list.append(anomalies)
    anomalies = [
        {"x": 0, "y": 0.5, "d": 0.2, "perm": 10},
        {"x": 0, "y": -0.5, "d": 0.2, "perm": 10},
        {"x": 0.5, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0, "d": 0.2, "perm": 10}
    ]
    anomalies_list.append(anomalies)

    anomalies = [
        {"x": 0, "y": 0, "d": 0.9, "perm": 10},
    ]
    anomalies_list.append(anomalies)

    # for anomalies in anomalies_list:
    #     reconstruction(n_el, anomalies, algo="gauss-newton", regularization=True, model=model, model2=model2, plot=True, use_vectorisation=True)
    #     break
    # plt.show()

    for n_el in [8, 16, 32, 64]:
        reconstructor = keras.models.load_model(f"model/eit_reconstruction_{n_el}pts.h5")
        autoencoder = keras.models.load_model(f"model/eit_auto_{n_el}pts.h5")
        model = Sequential([
            reconstructor,
            autoencoder
        ])
        for g in ["back-projection", "gauss-newton", "model"]:
            if g == "gauss-newton":
                regs = [False, True]
            else:
                regs = [False]
            for r in regs:
                t_total = datetime.timedelta(0)
                t_min = datetime.timedelta(days=999999)
                t_max = datetime.timedelta(0)
                m_total = 0
                m_min = np.inf
                m_max = 0
                k = 0
                for i in range(nb_sim):
                    for anomalies in anomalies_list:
                        t_taken, m_taken = reconstruction(n_el, anomalies, algo=g, model=model, regularization=r, plot=True)
                        t_total += t_taken
                        m_total += m_taken
                        if t_taken < t_min:
                            t_min = t_taken
                        if t_taken > t_max:
                            t_max = t_taken
                        if m_taken < m_min:
                            m_min = m_taken
                        if m_taken > m_max:
                            m_max = m_taken
                        k += 1
                        print(f'Progress: {(k/(nb_sim * len(anomalies_list)) * 100):.2f}%   {k}/{(nb_sim * len(anomalies_list))}')
                avg = t_total / k
                print(f'NB_elec = {n_el}. Algorithm = {g}, Regularization = {r}')
                print(f'Average time: {avg}')
                print(f'Minimum time: {t_min}')
                print(f'Maximum time: {t_max}')
                print(f'Average memory: {m_total / nb_sim}')
                print(f'Minimum memory: {m_min}')
                print(f'Maximum memory: {m_max}')
                df_list.append(pd.DataFrame([[n_el, f"{g}_reg_{r}", avg.total_seconds(), t_min.total_seconds(), t_max.total_seconds()]], columns=df_columns))
    df = pd.concat(df_list, ignore_index=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    print(df)


if __name__ == '__main__':
    main()
