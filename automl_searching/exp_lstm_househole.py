import sys

sys.path.insert(0, '../')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from matplotlib import pyplot as plt

from utils import HouseholdDataLoader, TSF_Data

dataload = HouseholdDataLoader(
    data_path=r"../../dataset/Household_power_consumption/household_power_consumption.txt")
data = dataload.data_by_hour

list_dataset = ['household', 'spain', 'cnu']
num_data = 0
result_path = list_dataset[num_data] + '/household_result_lstm'

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import Sequential
from keras.layers import Dense, LSTM


def build_model(tsf, output_width):
    model_tsf = Sequential()
    model_tsf.add(LSTM(200, return_sequences=True, activation='relu',
                       input_shape=(tsf.data_train[0].shape[1], 1)))
    model_tsf.add(LSTM(150))
    model_tsf.add(Dense(output_width))

    print(model_tsf.summary())
    model_tsf.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model_tsf


input_width = 168
callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
]

import numpy as np

for output_width in [60]:
    # Search model
    tsf = TSF_Data(data=data['Global_active_power'],
                   input_width=input_width,
                   output_width=output_width,
                   train_ratio=0.9)
    tsf.normalize_data(standardization_type=1)

    orig_stdout = sys.stdout
    f = open(result_path + f'/seaching_process_log_{str(output_width)}.txt', 'w')
    sys.stdout = f

    model_tsf = build_model(tsf, output_width)

    history = model_tsf.fit(x=tsf.data_train[0],
                            y=tsf.data_train[1],
                            epochs=100, validation_data=tsf.data_valid,
                            batch_size=32,
                            steps_per_epoch=100,
                            callbacks=callbacks)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    model_tsf.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                       verbose=2,
                       use_multiprocessing=True)
    sys.stdout = orig_stdout
    f.close()

    del model_tsf, tsf

    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('TCN after tunning')
    # plt.show()
    plt.savefig(result_path + "/" + str(output_width) + ".png", dpi=1200)
    plt.clf()
