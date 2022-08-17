import sys

sys.path.insert(0, '../')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from matplotlib import pyplot as plt

# Settings:
result_path = 'spain/spain_result'
list_stride = [24, 7]  # strides
kernel_size = 3  # kernel_size

import tensorflow as tf

layers = tf.keras.layers

from models import StrideDilationNetDetail

history_len = 168
input_width = history_len
output_width = 1
num_features = 1

from utils import TSF_Data, SpainDataLoader

dataloader = SpainDataLoader(data_path=r"../../dataset/Spain_Energy_Consumption")

import numpy as np

for output_width in [60]:
    orig_stdout = sys.stdout
    f = open(result_path + f'/T={history_len}-out={output_width}.txt', 'w')
    sys.stdout = f

    model = StrideDilationNetDetail(list_stride=list_stride,
                                    nb_filters=32,
                                    kernel_size=kernel_size,
                                    padding='causal',
                                    target_size=output_width,
                                    dropout_rate=0.0)

    input_test = layers.Input(shape=(input_width, num_features))

    model(input_test)
    model.summary()

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer='adam',
                  metrics=['mse', 'mae'])

    tsf = TSF_Data(data=dataloader.consumptions.loc[:, 20],
                   input_width=input_width,
                   output_width=output_width,
                   train_ratio=0.9)

    tsf.normalize_data()

    print('Train...')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=10,
                                                      mode='min')

    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    callbacks = [
        early_stopping,
        reduceLR
    ]

    history = model.fit(x=tsf.data_train[0],
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
    model.evaluate(tsf.data_test[0], tsf.data_test[1], batch_size=1,
                   verbose=2,
                   use_multiprocessing=True)

    sys.stdout = orig_stdout
    f.close()

    del model

    from matplotlib import pyplot as plt

    plt.plot(history.history['mse'][5:])
    plt.plot(history.history['val_mse'][5:])

    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('StrideDilatedNet')
    plt.savefig(result_path + "/" + f'{output_width}' + ".png", dpi=1200)
    plt.clf()
