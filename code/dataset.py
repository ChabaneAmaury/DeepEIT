from datetime import datetime

import numpy as np
import pandas as pd
from gast import literal_eval
from keras.layers import GaussianNoise
from matplotlib import pyplot as plt
from pyeit import mesh
from pyeit.mesh.shape import circle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
import tensorflow as tf


class DatasetDeepEIT:
    def __init__(self, nb_elect, batch_size):
        self.nb_elect = nb_elect
        self.batch_size = batch_size

        """ 0. construct mesh """
        # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
        self.mesh_obj, self.el_pos = mesh.create(nb_elect, h0=0.1, fd=circle)
        # extract node, element, alpha
        self.pts = self.mesh_obj["node"]
        self.tri = self.mesh_obj["element"]
        self.x, self.y = self.pts[:, 0], self.pts[:, 1]

    def load_data(self, csv_path, x: str, y: str):
        df = pd.read_csv(csv_path, usecols=[x, y], converters={x: literal_eval, y: literal_eval})

        x = df[x].tolist()
        y = df[y].tolist()

        return x, y

    def prepare(self, data: tuple, batch_size=32, stddev=0.1, seed=None, shuffle=False, augment=False):
        data_augmentation = Sequential([
            GaussianNoise(stddev=stddev, seed=seed),
        ])

        ds = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            ds = ds.shuffle(1000)

        # Batch all datasets.
        ds = ds.batch(batch_size)

        # Use data augmentation only on the training set.
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        ds = ds.cache()

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def get_datasets(self, x: str, y: str, batch_size, random_state=None):
        start_time = datetime.now()
        x, y = self.load_data("../dataset/eit_positive.csv", x=x, y=y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=random_state)

        ds_train = self.prepare((x_train, y_train),
                                batch_size=batch_size,
                                stddev=0.01,
                                seed=42,
                                shuffle=True,
                                augment=False)

        ds_test = self.prepare((x_test, y_test),
                               batch_size=batch_size,
                               stddev=0.1,
                               seed=42,
                               shuffle=True,
                               augment=False)

        print(f"Data preparation time: {datetime.now() - start_time}")

        return ds_train, ds_test

    def gauss_newtonian_extract_image(self, ds_n, grayscale=False):
        fig, ax = plt.subplots(1, 1, )

        fig.set_size_inches(2.56, 2.56)
        im = ax.tripcolor(self.x, self.y, self.tri, ds_n, shading="flat")
        # Image from plot
        ax.axis('off')
        fig.tight_layout(pad=0)
        # To remove the huge white borders
        ax.margins(0)

        fig.canvas.draw()
        if grayscale:
            # Grayscale
            image_from_plot = np.array(fig.canvas.buffer_rgba())
            image_from_plot = np.rint(image_from_plot[..., :3] @ [0.2126, 0.7152, 0.0722]).astype(np.uint8)
        else:
            # Color
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image_from_plot
