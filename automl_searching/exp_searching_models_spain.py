import sys

sys.path.insert(0, '../')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from utils import AreaEnergy, TSF_Data

from utils import SpainDataLoader

dataloader = SpainDataLoader(data_path=r"../../dataset/Spain_Energy_Consumption")

list_dataset = ['household', 'spain', 'cnu']
num_data = 1
result_path = list_dataset[num_data] + '/spain_result_auto'

import keras_tuner as kt
import os
import pandas as pd

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, LSTM
from keras import Sequential
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from tcn import TCN


def model_builder(hp):
    kernel_size = hp.Choice('kernel_size', values=[2, 3, 5, 7])
    nb_filters = hp.Choice('nb_filters', values=[16, 32, 64, 128])
    use_skip_connections = hp.Choice(
        'use_skip_connections', values=[True, False])

    use_batch_norm = hp.Choice(
        'use_batch_norm', values=[True, False])

    def temp(x): return 2 ** x

    def dilation_gen(x): return list(map(temp, range(x)))

    dilations = hp.Choice('dilations', values=list(range(2, 8)))
    # nb_stacks = hp.Choice('nb_stacks', values=[1, 2, 3, 4, 5])
    # nb_units_lstm = hp.Int('units_LSTM', min_value=32, max_value=320, step=32)

    x1 = TCN(input_shape=(input_width, 1),
             kernel_size=kernel_size,
             nb_filters=nb_filters,
             dilations=dilation_gen(dilations),
             use_skip_connections=use_skip_connections,
             use_batch_norm=use_batch_norm,
             use_weight_norm=False,
             use_layer_norm=False,
             return_sequences=False
             )(inputs)

    # x2 = LSTM(nb_units_lstm)(x1)

    x3 = Dense(units=tsf.data_train[1].shape[1])(x1)

    model_searching = Model(inputs, x3)

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    # hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model_searching.summary()

    model_searching.compile(loss=tf.keras.losses.Huber(),
                            optimizer='adam',
                            metrics=['mse', 'mae'])

    return model_searching


num_features = 1
max_trials = 20
input_width = 168

for output_width in [84, 96]:  # range(36, 73, 12):
    # Search model
    exp_path = "Spain_TCN_Tune/Bayesian/" + str(output_width) + "/"
    tuning_path = exp_path + "/models"

    if os.path.isdir(tuning_path):
        import shutil

        shutil.rmtree(tuning_path)

    tsf = TSF_Data(data=dataloader.consumptions.loc[:, 20],
                   input_width=input_width,
                   output_width=output_width,
                   train_ratio=0.9)

    tsf.normalize_data()
    input_width = tsf.data_train[0].shape[1]

    inputs = Input(shape=(input_width, num_features))

    print("[INFO] instantiating a random search tuner object...")

    tuner = kt.BayesianOptimization(
        model_builder,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=max_trials,
        seed=42,
        directory=tuning_path)

    # stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    orig_stdout = sys.stdout
    f = open(result_path + f'/seaching_process_log_{str(output_width)}.txt', 'w')
    sys.stdout = f

    tuner.search(tsf.data_train[0], tsf.data_train[1],
                 validation_data=tsf.data_valid,
                 callbacks=[tf.keras.callbacks.TensorBoard(exp_path + "/log")],
                 epochs=10)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Train real model_searching
    print(f"""
    kernel_size {best_hps.get('kernel_size')},  and
    nb_filters: {best_hps.get('nb_filters')}, 
    dilations: {best_hps.get('dilations')}
    use_batch_norm: {best_hps.get('use_batch_norm')}
    use_skip_connections: {best_hps.get('use_skip_connections')}
    """)

    # Train real model_searching

    # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
    model_best = tuner.hypermodel.build(best_hps)

    print('Train...')

    callbacks = [
        EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    ]

    history = model_best.fit(x=tsf.data_train[0],
                             y=tsf.data_train[1],
                             validation_data=tsf.data_valid,
                             epochs=100,
                             callbacks=[callbacks],
                             verbose=2,
                             use_multiprocessing=True)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    model_best.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                        verbose=2,
                        use_multiprocessing=True)
    sys.stdout = orig_stdout
    f.close()

    pd.DataFrame.from_dict(history.history).to_csv(result_path + '/history' + str(output_width) + '.csv', index=False)

    from matplotlib import pyplot as plt

    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('TCN after tunning')
    # plt.show()
    plt.savefig(result_path + "/" + str(output_width) + ".png", dpi=1200)
    plt.clf()

    del model_best
    del tuner, best_hps
