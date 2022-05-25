from keras.optimizer_v2.adam import Adam
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import BatchNormalization, Dense, Dropout, Reshape, Conv1D, MaxPooling1D, \
    Conv1DTranspose, Flatten, Conv2D, Conv2DTranspose
from tensorflow.python.keras.metrics import RootMeanSquaredError


def get_reconstructor(nb_outputs: int):
    """
    Generate the reconstruction model based on the given parameters.
    :param nb_outputs: The output shape in the form of an int.
    :return: The generated model.
    """

    model = Sequential([
        BatchNormalization(),
        Dense(500, activation="tanh"),
        Dropout(0.5),
        BatchNormalization(),
        Dense(nb_outputs, activation='relu')
    ], name="Reconstructor")

    optimizer = Adam(learning_rate=1e-3)
    model.compile(loss='mean_squared_error', optimizer=optimizer,
                  metrics=["mse", RootMeanSquaredError(), "mae"])

    return model


def get_autoencoder_denoiser(nb_outputs: int):
    """
    Generate the denoiser model based on the given parameters.
    :param nb_outputs: The output shape in the form of an int.
    :return: The generated model.
    """

    model = Sequential([
        Dense(1024, activation="relu"),
        Reshape((-1, 1)),
        Conv1D(16, 3, activation="tanh", strides=1, padding='same', kernel_regularizer='l1'),
        MaxPooling1D(2),
        Conv1D(8, 3, activation="tanh", strides=1, padding='same', kernel_regularizer='l1'),
        MaxPooling1D(2),

        Conv1D(8, 3, activation="tanh", strides=1, padding='same', kernel_regularizer='l1'),
        Conv1DTranspose(8, 3, activation="tanh", strides=2, padding='same', kernel_regularizer='l1'),
        Conv1D(16, 3, activation="tanh", strides=1, padding='same', kernel_regularizer='l1'),
        Conv1DTranspose(16, 3, activation="tanh", strides=2, padding='same', kernel_regularizer='l1'),
        Conv1DTranspose(1, 3, activation="tanh", padding='same', kernel_regularizer='l1'),
        Flatten(),
        Dense(nb_outputs, activation="relu"),
    ], name="Autoencoder_denoiser")

    model.compile(loss='mse', optimizer="adam",
                  metrics=["mse", RootMeanSquaredError(), "mae"])

    return model


def get_image_generator():
    model = Sequential([
        Dense(64 * 64 * 3, activation="relu"),
        Reshape((64, 64, 3)),
        Conv2D(8, 3, activation="relu", strides=1, padding="same", kernel_regularizer='l1'),
        Conv2DTranspose(8, 3, activation="relu", strides=2, padding="same", kernel_regularizer='l1'),
        Conv2D(16, 3, activation="relu", strides=1, padding="same", kernel_regularizer='l1'),
        Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same", kernel_regularizer='l1'),
        Conv2D(3, 3, activation="relu", padding="same", kernel_regularizer='l1'),
    ], name="Image_Generator")

    model.compile(loss='mae', optimizer="adam",
                  metrics=["mse", RootMeanSquaredError(), "mae"])

    return model


def fit_model(model, ds_train, ds_test, epochs, callbacks: list = None, verbose=True,
              plot_history=False):
    """
    Fits the given model based on given parameters.
    :param model: The model to train.
    :param ds_train: The training dataset.
    :param ds_test: The testing dataset.
    :param params: The model's parameters.
    :param callbacks: (Optional) Any given callback for debugging purposes.
    :param verbose: (Optional) The verbosity of the train steps.
    :param plot_history: (Optional) Choose to show or not the history of the training once it's finished.
    :return: The model, the history of the training and the evaluation score.
    """
    if callbacks is None:
        callbacks = []
    history = model.fit(ds_train, validation_data=ds_test,
                        epochs=epochs,
                        verbose=verbose,
                        workers=10,
                        use_multiprocessing=True,
                        max_queue_size=20,
                        callbacks=callbacks)
    score = model.evaluate(ds_test, verbose=verbose)

    if plot_history:
        # print(history.history.keys())
        fig, axs = plt.subplots(1, 2)
        # summarize history for accuracy
        axs[0].plot(history.history['mse'])
        axs[0].plot(history.history['val_mse'])
        axs[0].set_title('model mse')
        axs[0].set_ylabel('mse')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'test'])
        # summarize history for loss
        axs[1].plot(history.history['loss'])
        axs[1].plot(history.history['val_loss'])
        axs[1].set_title('model loss')
        axs[1].set_ylabel('loss')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'test'])
        # blank subplot
        # axs[-1, -1].axis('off')

    return model, history, score