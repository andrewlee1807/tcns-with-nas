import sys
from abc import ABC

sys.path.insert(0, '../')

import os
import pandas as pd
import argparse
from matplotlib import pyplot as plt

from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

import keras_tuner as kt
from models import StrideDilatedNet
from utils import AreaEnergy, TSF_Data, HouseholdDataLoader, SpainDataLoader, create_new_directory


def arg_parse(parser):
    parser.add_argument('--dataset_name', type=str, default='spain', help='Dataset Name: household; cnu; spain')
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='Dataset path')
    parser.add_argument('--history_len', type=int, default=168, help='History Length')
    parser.add_argument('--output_len', type=int, default=None, help='Prediction Length')
    parser.add_argument('--num_features', type=int, default=1, help='Number of features')
    parser.add_argument('--max_trials', type=int, default=20, help='Max trials')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--write_log_file', type=bool, default=True, help='CUDA Device')
    return parser.parse_args()


class HOModel(kt.HyperModel, ABC):
    def __init__(self, input_width, output_width, num_features):
        self.output_width = output_width
        self.input_width = input_width
        self.num_features = num_features

    def build(self, hp):
        kernel_size = hp.Choice('kernel_size', values=[2, 3, 5, 7])
        nb_filters = hp.Choice('nb_filters', values=[8, 16, 32, 64])
        dropout_rate = hp.Float('dropout_rate', 0, 0.5, step=0.1, default=0.5)
        layer_stride1 = hp.Choice('layer_stride1', values=range(1, 24))
        layer_stride2 = hp.Choice('layer_stride2', values=range(1, 7))

        model = StrideDilatedNet(list_stride=(layer_stride1, layer_stride2),
                                 nb_filters=nb_filters,
                                 kernel_size=kernel_size,
                                 padding='causal',
                                 target_size=self.output_width,
                                 dropout_rate=dropout_rate)

        # print model
        input_test = Input(shape=(self.input_width, self.num_features))
        model(input_test)
        model.summary()

        model.compile(loss=Huber(),
                      optimizer='adam',
                      metrics=['mse', 'mae'])

        return model


def auto_training(data_seq, dataset_name, history_len, output_len, num_features, max_trials, write_log_file):
    input_width = history_len
    num_features = num_features
    max_trials = max_trials
    result_path = f'{dataset_name}_auto'
    create_new_directory(result_path)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   mode='min')

    reduceLR = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)

    callbacks = [
        early_stopping,
        reduceLR
    ]

    if output_len is not None:
        output_widths = range(output_len, output_len + 1)
    else:
        output_widths = range(1, 25)

    for output_width in output_widths:
        # Search model
        exp_path = f"{dataset_name}_stride_Tune/Bayesian/" + str(output_width) + "/"
        tuning_path = exp_path + "/models"

        if os.path.isdir(tuning_path):
            import shutil

            shutil.rmtree(tuning_path)

        tsf = TSF_Data(data=data_seq,
                       input_width=input_width,
                       output_width=output_width,
                       train_ratio=0.9)

        tsf.normalize_data()

        model_builder = HOModel(input_width=input_width,
                                output_width=output_width,
                                num_features=num_features)

        tuner = kt.BayesianOptimization(
            model_builder,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=max_trials,
            seed=42,
            directory=tuning_path)

        if write_log_file:
            orig_stdout = sys.stdout
            f = open(result_path + f'/T={history_len}-out={output_width}.txt', 'w')
            sys.stdout = f

        tuner.search(tsf.data_train[0], tsf.data_train[1],
                     validation_data=tsf.data_valid,
                     callbacks=[TensorBoard(exp_path + "/log")],
                     epochs=10)

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model_best = tuner.hypermodel.build(best_hps)

        # Train real model_searching
        print(f"""
            kernel_size {best_hps.get('kernel_size')},  and
            nb_filters: {best_hps.get('nb_filters')}, 
            dropout_rate: {best_hps.get('dropout_rate')}
            layer_stride1: {best_hps.get('layer_stride1')}
            layer_stride2: {best_hps.get('layer_stride2')}
            """)

        print('Train...')

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

        if write_log_file:
            sys.stdout = orig_stdout
            f.close()

        pd.DataFrame.from_dict(history.history).to_csv(result_path + '/history' + str(output_width) + '.csv',
                                                       index=False)

        plt.plot(history.history['mse'][5:])
        plt.plot(history.history['val_mse'][5:])

        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.title('StrideDilatedNet after tuning')
        plt.savefig(result_path + "/" + f'{output_width}' + ".png", dpi=1200)
        plt.clf()

        del model_best
        del tuner, best_hps


def get_dataset(dataset_path, dataset_name):
    if dataset_name == 'cnu':
        공대7호관_HV_02 = AreaEnergy('공대7호관.HV_02',
                                 path_time=f"{dataset_path}/Electricity data_CNU/3.unit of time(일보)/")
        data = 공대7호관_HV_02.arr_seq_dataset
    elif dataset_name == 'spain':
        dataloader = SpainDataLoader(data_path=f"{dataset_path}/Spain_Energy_Consumption")
        data = dataloader.consumptions.loc[:, 20]
    elif dataset_name == 'household':
        dataloader = HouseholdDataLoader(
            data_path=f"{dataset_path}/Household_power_consumption/household_power_consumption.txt")
        data = dataloader.data_by_hour['Global_active_power']

    return data


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    args = arg_parse(argparse.ArgumentParser())
    # setup CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # Settings:
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    history_len = args.history_len
    num_features = args.num_features
    max_trials = args.max_trials
    output_len = args.output_len
    write_log_file = args.write_log_file

    data_seq = get_dataset(dataset_path, dataset_name)

    auto_training(data_seq, dataset_name, history_len, output_len, num_features, max_trials, write_log_file)


if __name__ == '__main__':
    main()
