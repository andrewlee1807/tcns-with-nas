import sys
sys.path.insert(0, '../')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from matplotlib import pyplot as plt


from utils import AreaEnergy, TSF_Data

공대7호관_HV_02 = AreaEnergy('공대7호관.HV_02',
                         path_time=r"../../dataset/Electricity data_CNU/3.unit of time(일보)/")

result_path = "cnu/cnu_result_gru" # saving the processing of training phase and images ploted
num_features = 1


from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Model, Sequential
from keras.layers import Dense, LSTM, Input, GRU, Dropout

def build_model(input_width, output_width):
    inputs = Input(shape=(input_width, num_features))
    # RNN Layers
    x1 = GRU(103, return_sequences=True, activation='relu',
                input_shape=(input_width, num_features))(inputs)
    x2 = GRU(103, return_sequences=False)(x1)
    x3 = Dropout(0.25)(x2)
    # Hidden dense layer
    x4 = Dense(64, activation='relu')(x3)
    x5 = Dropout(0.25)(x4)
    # Output layer
    x6 = Dense(output_width, activation='linear')(x5)

    model_tsf = Model(inputs, x6)

    model_tsf.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    
    print(model_tsf.summary())

    return model_tsf


input_width = 168
callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
]

import numpy as np

for output_width in [54, 66, 78, 56, 58, 62, 70, 80]:
    # Search model
    tsf = TSF_Data(data=공대7호관_HV_02.arr_seq_dataset,
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
    plt.title('GRU')
    # plt.show()
    plt.savefig(result_path + "/" + str(output_width) + ".png", dpi=120)
    plt.clf()