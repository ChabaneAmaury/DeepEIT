import datetime
import random
from datetime import datetime
from os.path import join

import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.models import load_model, clone_model
from keras.utils.vis_utils import plot_model
from matplotlib import pyplot as plt
from pyeit import mesh
from pyeit.eit import jac
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import circle
from sklearn.model_selection import train_test_split, KFold
import models
from dataset import DatasetDeepEIT


# from os import environ
# environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_nb_measurements(n):
    return n * (n - 3)


x_train = x_test = y_train = y_test = None

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

mseObject = MeanSquaredError()
tf.config.run_functions_eagerly(True)


def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.001)


scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

early_stop = EarlyStopping(monitor='val_loss', patience=100)


def gauss_newtonian(n_el, anomaly, regularization=False, model=None, model2=None, plot=True, raw_measures=False,
                    autoencoder=False, imager=False):
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
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

    if raw_measures:
        return f1.v

    start_time = datetime.now()
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
    print(ds_n.shape)
    print(f"Gauss-newtonian reconstruction: {datetime.now() - start_time}")

    if not plot:
        return ds_n
    else:
        if model is None and model2 is None:
            fig, axes = plt.subplots(1, 2, constrained_layout=True)
        elif model2 is None:
            fig, axes = plt.subplots(1, 3, constrained_layout=True)
        else:
            fig, axes = plt.subplots(1, 4, constrained_layout=True)

        fig.set_size_inches(6, 4)

        # plot ground truth
        ax = axes[0]
        ax.set_title('Ground truth')
        delta_perm = mesh_new["perm"] - mesh_obj["perm"]
        im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
        ax.set_aspect("equal")

        # plot EIT reconstruction
        ax = axes[1]
        ax.set_title(f'Gauss-newtonian reconstruction ({regularization=})')
        im = ax.tripcolor(x, y, tri, ds_n, shading="flat")
        ax.set_aspect("equal")

        if model is not None:
            """4. Model reconstruction"""
            start_time = datetime.now()
            if imager:
                pass
                # plot model reconstruction
                im1 = model.predict(np.array([f1.v]))[0]
                ax = axes[2]
                ax.set_title('New ML reconstruction')
                im = ax.imshow(im1)
                ax.set_aspect("equal")
            else:
                if autoencoder:
                    ds_n_model = model.predict(np.array([ds_n]))[0]
                else:
                    ds_n_model = model.predict(np.array([f1.v]))[0]

                # plot model reconstruction
                ax = axes[2]
                ax.set_title('New ML reconstruction')
                im = ax.tripcolor(x, y, tri, ds_n_model, shading="flat")
                ax.set_aspect("equal")
            print(f"New ML reconstruction: {datetime.now() - start_time}")

        if model2 is not None:
            start_time = datetime.now()
            """4.1. Second model reconstruction for comparison"""
            imager = False
            if imager:
                # plot model2 reconstruction
                im2 = model2.predict(np.array([f1.v]))[0]
                ax = axes[3]
                ax.set_title('Old ML reconstruction')
                im = ax.imshow(im2)
                ax.set_aspect("equal")
            else:
                if autoencoder:
                    ds_n_model2 = model2.predict(np.array([ds_n]))[0]
                else:
                    ds_n_model2 = model2.predict(np.array([f1.v]))[0]

                # plot model2 reconstruction
                ax = axes[3]
                ax.set_title('Old ML reconstruction')
                im = ax.tripcolor(x, y, tri, ds_n_model2, shading="flat")
                ax.set_aspect("equal")
            print(f"Old ML reconstruction: {datetime.now() - start_time}")

        fig.colorbar(im, ax=axes.ravel().tolist())


def kfold_crossvalidation(x, y, model_base, callbacks=None, verbose=0):
    if callbacks is None:
        callbacks = []

    # Define per-fold score containers
    acc_per_fold = []
    loss_per_fold = []

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(x, y):
        model = clone_model(model_base)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(loss='mean_squared_error', optimizer=optimizer,
                      metrics=["mse", RootMeanSquaredError(), "mae"])

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit(x[train], y[train],
                            validation_data=(x[test], y[test]),
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            workers=10,
                            use_multiprocessing=True,
                            max_queue_size=20,
                            callbacks=callbacks)

        # Generate generalization metrics
        scores = model.evaluate(x[test], y[test], verbose=0)
        print(
            # f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {
            # scores[1]}')
            ' '.join([f'Score for fold {fold_no}:'] + [f"{model.metrics_names[i]} of {scores[i]};" for i in
                                                       len(model.metrics_names)]))
        # f'Score for fold {fold_no}: {model.metrics_names[0]} of {history.history["loss"][-1]}; val_loss of {
        # history.history["val_loss"][-1]}')
        acc_per_fold.append(history.history["val_loss"][-1])
        loss_per_fold.append(history.history["loss"][-1])

        model.save(f"./model/eit_reconstruction_{nb_elect}pts_fold_{fold_no}.h5")

        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - MSE: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> MSE: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


def train(callbacks=None):
    if callbacks is None:
        callbacks = []
    ds_train, ds_test = deepDSObj.get_datasets(f"v{nb_elect}", f"delta_perm{nb_elect}", batch_size, random_state=42)

    input_shape = (ds_train.element_spec[0].shape[1],)
    output_shape = ds_train.element_spec[1].shape[1]

    model = models.get_reconstructor(nb_elect, output_shape)

    model, history, score = models.fit_model(model, ds_train, ds_test, epochs, verbose=True,
                                             plot_history=True, callbacks=callbacks)

    model.save(f"./model/eit_reconstruction_{nb_elect}pts_test.h5")
    return model


def model_evaluation(nb_elect, model1=None, model2=None, autoencoder=False, regularization=False, imager=False):
    rnge = 5
    for i in range(rnge):
        anomalies = []
        for _ in range(random.randint(1, 5)):
            anomalies.append(
                {"x": random.uniform(-1.0, 1.0), "y": random.uniform(-1.0, 1.0), "d": random.uniform(0.0, 0.5),
                 "perm": random.uniform(1.0, 10.0)})
        gauss_newtonian(nb_elect, anomalies, regularization=regularization, model=model1, model2=model2, plot=True,
                        autoencoder=autoencoder,
                        imager=imager)
        print(f"{i + 1}/{rnge}")

    # Tests used in the paper
    for anomalies in anomalies_list:
        gauss_newtonian(nb_elect, anomalies, regularization=regularization, model=model1, model2=model2, plot=True,
                        autoencoder=autoencoder,
                        imager=imager)
        # break

    plt.show()


def main():
    model1 = train([early_stop, scheduler_callback])
    model2 = load_model(f"./model/eit_reconstruction_{nb_elect}pts.h5")
    model_evaluation(nb_elect, model1, model2)


def main_kfold():
    start_time = datetime.now()
    x, y, = deepDSObj.load_data("./dataset/eit_positive.csv", x=f"v{nb_elect}", y=f"delta_perm{nb_elect}")
    print(f"Data preparation time: {datetime.now() - start_time}")

    model = models.get_reconstructor(nb_elect, y.shape[1])

    kfold_crossvalidation(x, y, model, verbose=1, callbacks=[scheduler_callback, early_stop])


def main_autoencoder():
    start_time = datetime.now()

    x, y = deepDSObj.load_data("./dataset/eit_positive.csv", x=f"v{nb_elect}", y=f"delta_perm{nb_elect}")

    reconstruction = load_model(f"./model/eit_reconstruction_{nb_elect}pts")
    x = reconstruction.predict(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    ds_train = deepDSObj.prepare((x_train, y_train),
                                 batch_size=16,
                                 stddev=0.01,
                                 seed=42,
                                 shuffle=True,
                                 augment=False)

    ds_test = deepDSObj.prepare((x_test, y_test),
                                batch_size=16,
                                stddev=0.1,
                                seed=42,
                                shuffle=True,
                                augment=False)

    print(f"Data preparation time: {datetime.now() - start_time}")

    model = models.get_autoencoder_denoiser(nb_elect, x.shape[-1])

    model.fit(ds_train, epochs=50, callbacks=[early_stop],
              validation_data=ds_test, max_queue_size=10, workers=1, use_multiprocessing=True)

    model.save(f"./model/eit_auto_{nb_elect}pts.h5")


def main_image_generator():
    start_time = datetime.now()

    _, ds_x = deepDSObj.load_data("./dataset/eit_positive.csv", x=f"ds_n{nb_elect}", y=f"delta_perm{nb_elect}")

    # ds_x, x1 = ds_x + x1, None

    def load_images(path):
        import glob
        filelist = glob.glob(join(path, f"delta_perm{nb_elect}/*.png"))
        return np.array([np.array(Image.open(fname))[:, :, :3] for fname in filelist])

    ds_y = load_images("./dataset/images")
    ds_x, ds_y = np.array(ds_x), (np.array(ds_y, dtype=np.uint8) / 255.0).astype("float32")

    x_train, x_test, y_train, y_test = train_test_split(ds_x, ds_y, test_size=0.20, random_state=42)

    ds_train = deepDSObj.prepare((x_train, y_train),
                                 batch_size=64,
                                 stddev=0.01,
                                 seed=42,
                                 shuffle=True,
                                 augment=False)

    ds_test = deepDSObj.prepare((x_test, y_test),
                                batch_size=64,
                                stddev=0.1,
                                seed=42,
                                shuffle=True,
                                augment=False)

    print(f"Data preparation time: {datetime.now() - start_time}")

    activation = "relu"

    model = models.get_image_generator()

    model.fit(ds_train, epochs=10, callbacks=[early_stop],
              validation_data=ds_test)

    model.save(f"./model/eit_image_{nb_elect}pts.h5")


def main_model_test():
    reconstruction = load_model(f"./model/eit_reconstruction_{nb_elect}pts.h5")
    autoencoder = load_model(f"./model/eit_auto_{nb_elect}pts.h5")
    # imager = load_model(f"model/eit_image_{nb_elect}pts")

    model = Sequential([
        reconstruction,
        autoencoder,
        # imager
    ])
    model2 = Sequential([
        reconstruction,
        # autoencoder
    ])
    model.build((None, get_nb_measurements(nb_elect)))
    plot_model(model, to_file="./model.png", expand_nested=True, show_shapes=True, show_layer_names=False)
    model_evaluation(nb_elect, model, model2, regularization=False, imager=False)


def evaluate_model():
    reconstruction = load_model(f"./model/eit_reconstruction_{nb_elect}pts.h5")
    autoencoder = load_model(f"./model/eit_auto_{nb_elect}pts.h5")
    # imager = load_model(f"model/eit_image_{nb_elect}pts")

    _, ds_test = deepDSObj.get_datasets(f"v_{nb_elect}", f"delta_perm{nb_elect}", batch_size, random_state=42)

    model = Sequential([
        reconstruction,
        autoencoder,
        # imager
    ])

    model.compile(loss='mean_squared_error', optimizer="adam",
                  metrics=["mse", RootMeanSquaredError(), "mae"])

    score = model.evaluate(ds_test, verbose=True)


nb_elect = 32
batch_size = 32
epochs = 100
num_folds = 10

deepDSObj = DatasetDeepEIT(nb_elect=nb_elect,
                           batch_size=batch_size)

if __name__ == "__main__":
    main()
    # main_kfold()
    # main_autoencoder()
    # main_image_generator()
    # main_model_test()
    # evaluate_model()
