import sys
sys.path.insert(0, '/home/andrew/Time Series/TSDatasets')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from matplotlib import pyplot as plt

from utils import SpainDataLoader, TSF_Data

dataloader = SpainDataLoader(data_path="/home/andrew/Time Series/dataset/Spain_Energy_Consumption")

result_path = "spain_result_lstm" # saving the processing of training phase and images ploted
num_features = 1


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Model
from keras.layers import Dense, LSTM, Input

def build_model(input_width, output_width):
    inputs = Input(shape=(input_width, num_features))
    x1 = LSTM(200, return_sequences=True, activation='relu',
                input_shape=(input_width, 1))(inputs)
    x2 = LSTM(150)(x1)
    x3 = Dense(output_width)(x2)

    model_tsf = Model(inputs, x3)

    print(model_tsf.summary())
    model_tsf.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model_tsf


input_width = 168
callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
]

import numpy as np

for output_width in [60, 84]:
    # Search model
    tsf = TSF_Data(data=dataloader.consumptions.loc[:, 20],
                input_width=input_width,
                output_width=output_width,
                train_ratio=0.9)

    tsf.normalize_data()

    orig_stdout = sys.stdout
    f = open(result_path + f'/seaching_process_log_{str(output_width)}.txt', 'w')
    sys.stdout = f

    model_tsf= build_model(input_width, output_width)

    history = model_tsf.fit(x=tsf.data_train[0],
                            y=tsf.data_train[1],
                            epochs=100, 
                            validation_data=tsf.data_valid,
                            batch_size=32,
                            verbose=2,
                            steps_per_epoch=100,
                            callbacks=callbacks)

    print("=============================================================")
    print("Minimum val mse:")
    print(min(history.history['val_mse']))
    print("Minimum training mse:")
    print(min(history.history['mse']))
    model_tsf.evaluate(tsf.data_test[0],tsf.data_test[1], batch_size=1,
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
    plt.title('LSTM')
    # plt.show()
    plt.savefig(result_path + "/" + str(output_width) + ".png", dpi=1200)
    plt.clf()