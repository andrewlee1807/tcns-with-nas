# Convert data to float
def check_error_convert_float(col):
    """
    :param col: list or series of values
    :return:
    """
    num = 0
    for i in col:
        try:
            float(i)
        except Exception as e:
            num += 1
            print(e)
    print("Number of error data: ", num)


def create_new_directory(path, delete_existed=False):
    import os
    # delete/ rename the path directory
    if os.path.isdir(path):
        if delete_existed:
            # delete
            import shutil
            shutil.rmtree(path)
        else:
            # rename
            import time
            new_name = f"{path}_{time.time()}"
            os.rename(path, new_name)
    os.mkdir(path)


import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt


# read configuration
def read_config():
    import yaml

    # read yaml file
    config_path = '/home/dspserver/andrew/TSDatasets/config.yaml'
    with open(config_path, encoding='cp437') as file:
        config = yaml.safe_load(file)
        # print(config)
    return config


# config = read_config()


class TSF:
    """
    Time Series Forcasting
    Purpose: Split to Train, Test, Val from a raw sequence dataset
    Params:
        :data(sequence dataset),
        :input_width(input's length),
        :label_width(label's length)
    Example:
        seq = np.asarray(list((range(1000))))

        tsf = TSF(seq, 5, 1, 1)

        for (x, y) in tsf.train:

            print(x.numpy(), y.numpy())

            #OUTPUT: [[0 1 2 3 4 5 6]
                      [1 2 3 4 5 6 7]
                      [2 3 4 5 6 7 8]] [7 8 9]


    """

    def __init__(self, data, input_width: int, label_width: int, shift=1, batch_size=32, ratio=0.7, shuffle=True):
        # TODO: saving time sequence
        # Sequence data type without split 2 sets (input, output)
        self.test_data = None
        self.val_data = None
        self.train_data = None

        # Work out the window parameters.
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + label_width
        # INPUT
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        # OUTPUT
        self.labels_slice = slice(self.input_width, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        # Store the raw data.
        self.raw_data = np.array(data, dtype=np.float32)
        self.number_feature = self.raw_data.ndim
        self.split_data(ratio)
        self.normalize_data()
        # Split the data into INPUT and OUTPUT
        # self.train = self.make_dataset(self.train_data)
        # print(f"Training data generated {len(self.train_data)} sequences")
        # self.test = self.make_dataset(self.test_data)
        # print(f"Testing data generated {len(self.test_data)} sequences")
        # self.val = self.make_dataset(self.val_data)
        # print(f"Validation data generated {len(self.val_data)} sequences")

        # Test
        print(f'Inputs shape (batch, time, features): {self.example[0].shape}')
        print(f'Labels shape (batch, time, features): {self.example[1].shape}')

        # for example_inputs, example_labels in self.train.take(1):
        #     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        #     print(f'Labels shape (batch, time, features): {example_labels.shape}')

    @property
    def train(self):
        return self.make_dataset(self.train_data)

    @property
    def test(self):
        return self.make_dataset(self.test_data)

    @property
    def val(self):
        return self.make_dataset(self.val_data)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def split_data(self, ratio):
        n = len(self.raw_data)
        val_ratio = 2 * (1 - ratio) / 3
        self.train_data = self.raw_data[:int(n * ratio)]
        self.val_data = self.raw_data[int(n * ratio):int(n * (ratio + val_ratio))]
        self.test_data = self.raw_data[int(n * (ratio + val_ratio)):]

    def __split_window(self, features):
        # batch, time, features
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        if len(data.shape) == 1:  # incase univariate feature
            data = np.array(data, dtype=np.float32)[..., np.newaxis]

        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.shift,
            shuffle=self.shuffle,
            batch_size=self.batch_size)

        # print(ds.take(1))

        return ds.map(self.__split_window)

    def normalize_data(self):
        """The mean and standard deviation should only be computed using the training data so that the models
        have no access to the values in the validation and test sets."""
        from sklearn.preprocessing import MinMaxScaler
        scaler_train = MinMaxScaler()
        self.train_data = scaler_train.fit_transform(self.train_data)
        self.val_data = scaler_train.fit_transform(self.val_data)
        self.test_data = scaler_train.fit_transform(self.test_data)

        # train_mean = self.train_data.mean()
        # train_std = self.train_data.std()
        #
        # self.train_data = (self.train_data - train_mean) / train_std
        # self.val_data = (self.val_data - train_mean) / train_std
        # self.test_data = (self.test_data - train_mean) / train_std

    def export_label_array(self, data_pack):
        y_label = np.empty((0)) if self.label_width == 1 else np.empty((0, self.label_width))
        for (_, i) in data_pack:
            y_label = np.concatenate([y_label, np.squeeze(i.numpy())])
        return y_label


from sklearn.model_selection import train_test_split


class TSF_Data:
    """
    This class only support to prepare training (backup to TSF class)
    example:
    tsf = TSF_Data(data=공대7호관_HV_02.arr_seq_dataset,
                input_width=21,
                output_width=1,
                train_ratio=0.9)
    tsf.normalize_data(standardization_type=1)
    """

    def __init__(self, data, input_width: int, output_width: int, shift=1, batch_size=32, train_ratio=None,
                 shuffle=False):
        """
        return:
        data_train,
        data_valid,
        data_test,
        function: inverse_scale_transform
        """
        self.data_train = None
        self.data_test = None
        self.scaler_x = None
        self.scaler_y = None
        self.raw_data = data
        self.input_width = input_width
        self.output_width = output_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.split_data(train_ratio)
        self.data_train = self.build_tsd(self.X_train)
        self.data_valid = self.build_tsd(self.X_valid)
        if self.X_test is not None:
            self.data_test = self.build_tsd(self.X_test)
        else:
            self.data_test = None

        # self.normalize_data()

    def split_data(self, train_ratio):
        self.X_test = None  # No testing, using whole data to train
        X_train = self.raw_data
        if train_ratio is not None:
            X_train, self.X_test = train_test_split(
                self.raw_data, train_size=train_ratio, shuffle=self.shuffle)
        self.X_train, self.X_valid = train_test_split(
            X_train, train_size=0.9, shuffle=self.shuffle)

    def inverse_scale_transform(self, y_predicted):
        """
        un-scale predicted output 
        """
        if self.scaler_y is not None:
            return self.scaler_y.inverse_transform(y_predicted)
        return y_predicted

    def normalize_data(self, standardization_type=1):
        """The mean and standard deviation should only be computed using the training data so that the models
        have no access to the values in the validation and test sets.
        1: MinMaxScaler, 2: StandardScaler, 3: RobustScaler, 4: PowerTransformer
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
        standardization_methods = {1: MinMaxScaler, 2: StandardScaler, 3: RobustScaler, 4: PowerTransformer}
        standardization_method = standardization_methods[standardization_type]
        scaler_x = standardization_method()
        scaler_x.fit(self.data_train[0])
        scaler_y = standardization_method()
        scaler_y.fit(self.data_train[1])
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        # self.data_train = scaler_x.transform(
        #     self.data_train[0]), scaler_y.transform(self.data_train[1])
        # # converting into L.S.T.M format
        # self.data_train = self.data_train[0], self.data_train[1]
        # self.data_valid = scaler_x.transform(
        #     self.data_valid[0]), scaler_y.transform(self.data_valid[1])
        # self.data_valid = self.data_valid[0], self.data_valid[1]
        # if self.data_test is not None:
        #     self.data_test = scaler_x.transform(
        #         self.data_test[0]), scaler_y.transform(self.data_test[1])
        #     self.data_test = self.data_test[0], self.data_test[1]
        self.data_train = scaler_x.transform(
            self.data_train[0]), scaler_y.transform(self.data_train[1])
        # converting into L.S.T.M format
        self.data_train = self.data_train[0][...,
                                             np.newaxis], self.data_train[1]
        self.data_valid = scaler_x.transform(
            self.data_valid[0]), scaler_y.transform(self.data_valid[1])
        self.data_valid = self.data_valid[0][...,
                                             np.newaxis], self.data_valid[1]
        if self.data_test is not None:
            self.data_test = scaler_x.transform(
                self.data_test[0]), scaler_y.transform(self.data_test[1])
            self.data_test = self.data_test[0][...,
                                               np.newaxis], self.data_test[1]

    def build_tsd(self, data):
        X_data, y_label = [], []
        if self.input_width >= len(data) - self.output_width - 168:
            raise ValueError(
                f"Cannot devide sequence with length={len(data)}. The dataset is too small to be used input_length= {self.input_width}. Please reduce your input_length")

        for i in range(self.input_width, len(data) - self.output_width):
            X_data.append(data[i - self.input_width:i])
            y_label.append(data[i:i + self.output_width])

        X_data, y_label = np.array(X_data), np.array(y_label)

        return X_data, y_label


class AreaEnergy:
    """
    Ex: 공대7호관_HV_02 = AreaEnergy('공대7호관.HV_02')
    """

    def __init__(self, name_area, path_time=None):
        import datetime
        import os
        self.name_area = name_area
        self.df_seq_dataset = []
        self.arr_seq_dataset = []

        if path_time is None:
            path_time = config['dataset']['cnu_path']
        # List files in hours per a days with years
        list_path_dataset = [path_time + '/' + i for i in os.listdir(path_time)]
        list_path_dataset.sort()
        time_index = [i.split('/')[-1].split('.')[0] for i in list_path_dataset]
        columns_name = ['차단기 명'] + [str(i) for i in range(1, 25)]
        # Generate the time index
        index_datatime = np.array(list(
            map(lambda x: [datetime.datetime(int(x[:4]), int(x[4:6]), int(x[6:8]), i) for i in range(24)],
                time_index))).flatten()

        self.index_all_time = index_datatime

        for file_path in list_path_dataset:
            df = pd.read_excel(file_path, engine='xlrd')
            df = df.loc[3:]
            self.extract_observation(df)

    def extract_observation(self, df):
        columns_name = ['차단기 명'] + [str(i) for i in range(1, 25)]
        # print(columns_name)
        df_sub = df.loc[:, ~df.columns.isin(['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 27'])]
        df_sub.columns = columns_name
        df_sub = df_sub.reset_index().drop(columns=['index'])
        record = df_sub[df_sub['차단기 명'] == self.name_area]
        self.arr_seq_dataset = np.append(self.arr_seq_dataset, np.squeeze(record.to_numpy())[1:])

    def plot_sequence(self, df_seq=None):
        import matplotlib.font_manager as fm
        path_pen = 'C:/Windows/Fonts/BatangChe.TTF'
        font = fm.FontProperties(fname=path_pen, size=15)

        if df_seq is None:
            df_seq = pd.DataFrame(self.arr_seq_dataset, columns=['전력사용량'], index=self.index_all_time)
            self.df_seq_dataset = df_seq
        fig, axs = plt.subplots(figsize=(25, 8))
        plt.title('전력사용량 of ' + self.name_area + ' [kWh]', fontproperties=font)
        plot_features = df_seq['전력사용량']
        plot_features.index = df_seq.index
        _ = plot_features.plot(subplots=True)
        plt.show()


"""Spain Data Process """


def fix_DST(data):
    data = data[~data.index.duplicated(keep='first')]
    data = data.resample('H').ffill()
    return data


def crop(data):
    hour_index = data.index.hour
    t0 = data[hour_index == 0].head(1).index
    tn = data[hour_index == 23].tail(1).index
    data.drop(data.loc[data.index < t0[0]].index, inplace=True)
    data.drop(data.loc[data.index > tn[0]].index, inplace=True)
    return


class SpainDataLoader:

    def __init__(self, data_path=None):
        self.weather_forecast = None
        self.weather = None
        self.consumptions_scaled = None
        self.consumptions = None
        self.data_path = data_path
        if self.data_path is None:
            self.data_path = config['dataset']['spain_path']
        self.categories = ['consumption', 'weather', 'profiles']
        self.files = [self.data_path + '/' + '20201015_' + name + '.xlsx' for name in self.categories]
        self.load_data()

    def scale_data(self, data):
        x = data.groupby(data.index.date).mean()
        x.index = pd.to_datetime(x.index)
        x = x.append(pd.DataFrame(x.tail(1), index=x.tail(1).index + pd.Timedelta(days=1)))
        x = x.resample('h').ffill()[:-1]
        x.index = data.index
        y = data / x
        return y

    def load_metadata(self):
        customers = pd.read_excel(self.files[self.categories.index('profiles')])
        customers.columns = ['customer', 'profile']
        profiles = pd.DataFrame(customers['profile'].unique(), columns=['profile'])
        # holidays = hd.ES(years=list(range(2010, 2021)), prov="MD")
        return customers, profiles

    def load_data(self):
        consumptions = pd.read_excel(self.files[self.categories.index('consumption')], parse_dates=[0], index_col=0)
        consumptions.columns = pd.DataFrame(consumptions.columns, columns=['customer']).index
        consumptions.index.name = 'time'
        consumptions = fix_DST(consumptions)
        crop(consumptions)
        self.consumptions = consumptions
        consumptions_scaled = self.scale_data(consumptions)
        weather = pd.read_excel(self.files[self.categories.index('weather')], parse_dates=[0], index_col=0)
        weather.columns = consumptions.columns
        weather.index.name = 'time'
        weather = fix_DST(weather)
        weather_forecast = weather.copy()
        weather_forecast.index = weather.index - pd.Timedelta(days=1)
        crop(weather)
        crop(weather_forecast)
        self.consumptions_scaled = consumptions_scaled
        self.weather = weather
        self.weather_forecast = weather_forecast
        # return consumptions, consumptions_scaled, weather, weather_forecast

    def prepare_data(self, consumptions, weather, holidays):
        """For calculating the means"""
        days = pd.DataFrame(pd.to_datetime(consumptions.index.date), index=consumptions.index, columns=['date'])
        days['day_of_week'] = list(days.index.dayofweek)
        days['day_of_month'] = list(days.index.day)
        days['month'] = list(days.index.month)
        days['day_category'] = days['day_of_week'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 4})
        days.loc[days['date'].apply(lambda d: d in holidays), 'day_category'] = 4
        days = days.groupby(['date']).first()
        consumptions_daily_mean = pd.DataFrame(consumptions.groupby(consumptions.index.date).mean(), index=days.index)
        weather_daily_mean = pd.DataFrame(weather.groupby(weather.index.date).mean(), index=days.index)
        return consumptions_daily_mean, weather_daily_mean, days

    def plot_sequence(self, customer=20):
        fig, axs = plt.subplots(figsize=(25, 8))
        _ = self.consumptions.loc[:, customer].plot(color='purple', alpha=0.7,
                                                    ylabel='energy consumption in kWh per hour',
                                                    title='Consumption of Customer #' + str(customer))
        plt.show()


"""==================================================="""

"""Household Electricity """


def fill_missing(data):
    one_day = 23 * 60
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = data[row - one_day, col]


class HouseholdDataLoader:
    def __init__(self, data_path=None):
        self.df = None
        self.data_by_days = None
        self.data_by_hour = None
        self.data_path = data_path
        if self.data_path is None:
            self.data_path = config['dataset']['household_path']
        self.load_data()

    def load_data(self):

        df = pd.read_csv(self.data_path, sep=';',
                         parse_dates={'dt': ['Date', 'Time']},
                         infer_datetime_format=True,
                         low_memory=False, na_values=['nan', '?'],
                         index_col='dt')

        droping_list_all = []
        for j in range(0, 7):
            if not df.iloc[:, j].notnull().all():
                droping_list_all.append(j)
        for j in range(0, 7):
            df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

        fill_missing(df.values)

        self.df = df
        self.data_by_days = df.resample('D').sum()  # all the units of particular day
        self.data_by_hour = df.resample('H').sum()  # all the units of particular day
