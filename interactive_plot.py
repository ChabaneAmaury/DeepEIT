import random

import numpy as np
from keras.models import load_model, Sequential
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from pyeit import mesh
from pyeit.eit.fem import Forward
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import circle
from tensorflow import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

x_train = x_test = y_train = y_test = None
ex_mat = None
fwd = None

def gauss_newtonian(n_el, anomaly, model=None, model2=None):
    global ex_mat, fwd
    vmin, vmax = 0, 10

    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(n_el, el_dist)

    """ 0. construct mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(n_el, h0=0.1, fd=circle)
    fwd = Forward(mesh_obj, el_pos)

    def init(anomaly):
        # extract node, element, alpha
        pts = mesh_obj["node"]
        tri = mesh_obj["element"]
        x, y = pts[:, 0], pts[:, 1]

        """ 1. problem setup """
        anomaly = anomaly
        mesh_obj["alpha"] = np.random.rand(tri.shape[0]) * 200 + 100
        mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)
        return x, y, tri, mesh_new

    x, y, tri, mesh_new = init(anomaly)

    def simulate(mesh_new):
        """ 2. FEM simulation """
        # el_dist, step = 1, 1
        # ex_mat = eit_scan_lines(n_el, el_dist)

        # calculate simulated data
        # fwd = Forward(mesh_obj, el_pos)
        # f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"], use_vectorization=True)
        f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"], use_vectorization=True)
        return f1

    f1 = simulate(mesh_new)

    if model2 is None:
        fig, axes = plt.subplots(1, 2, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, constrained_layout=True)

    fig.set_size_inches(6, 4)

    # plot ground truth
    ax = axes[0]
    ax.set_title('Ground truth')
    delta_perm = mesh_new["perm"] - mesh_obj["perm"]
    im_g = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
    ax.set_aspect("equal")

    if model is not None:
        """4. Model reconstruction"""
        ds_n_model = model.predict(np.array([f1.v]))[0]

        # plot model reconstruction
        ax = axes[1]
        ax.set_title('New ML reconstruction')
        im_m1 = ax.tripcolor(x, y, tri, ds_n_model, shading="flat")
        ax.set_aspect("equal")

    if model2 is not None:
        """4.1. Second model reconstruction for comparison"""
        ds_n_model2 = model2.predict(np.array([f1.v]))[0]

        # plot model2 reconstruction
        ax = axes[2]
        ax.set_title('Old ML reconstruction')
        im_m2 = ax.tripcolor(x, y, tri, ds_n_model2, shading="flat")
        ax.set_aspect("equal")

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)

    fig.colorbar(im_m1, ax=axes.ravel().tolist())

    # Make a horizontal slider to control the X.
    ax_x = plt.axes([0.25, 0.15, 0.65, 0.03])
    x_slider = Slider(
        ax=ax_x,
        label='X',
        valmin=-1,
        valmax=1,
        valinit=anomaly[0]['x'],
    )

    # Make a horizontal slider to control the perm.
    ax_perm = plt.axes([0.25, 0.1, 0.65, 0.03])
    perm_slider = Slider(
        ax=ax_perm,
        label='Perm',
        valmin=0,
        valmax=11,
        valinit=anomaly[0]['perm'],
    )

    # Make a vertically oriented slider to control the Y
    ax_y = plt.axes([0.1, 0.25, 0.0225, 0.63])
    y_slider = Slider(
        ax=ax_y,
        label="Y",
        valmin=-1,
        valmax=1,
        valinit=anomaly[0]["y"],
        orientation="vertical"
    )

    # Make a vertically oriented slider to control the diameter
    ax_d = plt.axes([0.15, 0.25, 0.0225, 0.63])
    d_slider = Slider(
        ax=ax_d,
        label="Diameter",
        valmin=0,
        valmax=1,
        valinit=anomaly[0]["d"],
        orientation="vertical"
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        anomaly[0]['x'] = x_slider.val
        anomaly[0]['y'] = y_slider.val
        anomaly[0]['d'] = d_slider.val
        anomaly[0]['perm'] = perm_slider.val
        x, y, tri, mesh_new = init(anomaly)
        f1 = simulate(mesh_new)
        delta_perm = mesh_new["perm"] - mesh_obj["perm"]
        im_g.set_array(np.real(delta_perm))

        ds_n_model = model.predict(np.array([f1.v]))[0]
        im_m1.set_array(ds_n_model)

        if model2 is not None:
            ds_n_model = model2.predict(np.array([f1.v]))[0]
            im_m2.set_array(ds_n_model)

        fig.canvas.draw_idle()

    # register the update function with each slider
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    d_slider.on_changed(update)
    perm_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', hovercolor='0.975')

    def reset(event):
        x_slider.reset()
        y_slider.reset()
        d_slider.reset()
        perm_slider.reset()

    button.on_clicked(reset)
    plt.show()


def main_interact(nb_elect):
    model1 = None
    # model1 = load_model(f"model/eit_reconstruction_{nb_elect}pts_500")
    reconstruction = load_model(f"model/eit_reconstruction_{nb_elect}pts")
    autoencoder = load_model(f"model/eit_auto_{nb_elect}pts")

    model1 = Sequential([
        reconstruction,
        autoencoder,
    ])
    anomalies = [{"x": random.uniform(-1.0, 1.0), "y": random.uniform(-1.0, 1.0), "d": random.uniform(0.0, 0.5),
                  "perm": 10}]
    gauss_newtonian(nb_elect, anomalies, model=model1)


if __name__ == "__main__":
    main_interact(nb_elect=64)
