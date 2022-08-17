from matplotlib import pyplot as plt
import os
import glob

import pandas as pd

import numpy as np

from natsort import os_sorted

list_dataset = ['household', 'spain', 'cnu']
markers = ['.', 'o', '*', '+', 'x', '^', "v", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", ",", "h", "H", "X", "D",
           "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
           ]
type_displays = ["MSE", "MAE"]

keywords = "Total params"


def export_keywords_value_from_txt(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()

    find_params_line = lambda x: keywords in lines[x]
    params_line_index = list(filter(find_params_line, range(len(lines) - 1, -1, -1)))[0]
    params_value = float(lines[params_line_index].split(' ')[-1].replace('\n', "").replace(",", ""))
    return params_value


def export_mse_mae_from_txt(pth):
    with open(pth, 'r') as f:
        last_line = f.readlines()[-1]
    last_line = last_line.split(" ")
    find_mse_index = lambda x: "mse" in last_line[x]
    find_mae_index = lambda x: "mae" in last_line[x]

    mse_index = list(filter(find_mse_index, range(len(last_line))))[0] + 1
    mae_index = list(filter(find_mae_index, range(len(last_line))))[0] + 1
    mae = float(last_line[mae_index])
    mse = float(last_line[mse_index])
    return mse, mae


def list_error(listdir):
    listdir = os_sorted(listdir)
    list_mse = []
    list_mae = []
    list_order = []
    for filename in listdir:
        mse, mae = export_mse_mae_from_txt(filename)
        list_mae.append(mae)
        list_mse.append(mse)
        # extend to get detail order
        try:
            hour = int(filename[-6:-4])
            list_order.append(hour)
        except Exception:
            try:
                hour = int(filename[-5:-4])
                list_order.append(hour)
            except Exception:
                pass
    return list_mse, list_mae, list_order


def plot_data_error(type_display, dataset_order, lstm_err, our_err, gru_err):
    fig, ax = plt.subplots()
    length_max = max(len(lstm_err), len(our_err), len(gru_err))
    ax.plot(list(range(1, 25)), lstm_err[:length_max],
            marker='.', linestyle='-', linewidth=0.5, label='lstm')

    ax.plot(list(range(1, 25)), our_err[:length_max],
            marker='o', markersize=8, linestyle='-', label='our')

    ax.plot(list(range(1, 25)), gru_err[:length_max],
            marker='*', markersize=8, linestyle='-', label='gru')

    ax.set_ylabel(type_display + f" on Dataset {dataset_order + 1} test set")
    ax.legend()
    plt.savefig(type_display + f" {list_dataset[dataset_order]}.png", dpi=120)
    plt.clf()


def compare_tcn_auto():
    for dataset_order in range(0, len(list_dataset)):
        # LSTM
        path_folder1 = f"automl_searching/{list_dataset[dataset_order]}_result_lstm/*.txt"
        listdir = glob.glob(path_folder1)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        lstm_errs = list_error(listdir)

        # TCN auto-generated search
        path_folder2 = f"automl_searching/{list_dataset[dataset_order]}_result_auto/*.txt"
        listdir = glob.glob(path_folder2)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        our_errs = list_error(listdir)

        #  GRU
        path_folder3 = f"automl_searching/{list_dataset[dataset_order]}_result_gru/*.txt"
        listdir = glob.glob(path_folder3)
        listdir.sort(key=lambda x: os.path.getmtime(x))
        gru_errs = list_error(listdir)

        plot_data_error("MSE", dataset_order, lstm_errs[0], our_errs[0], gru_errs[0])
        plot_data_error("MAE", dataset_order, lstm_errs[1], our_errs[1], gru_errs[1])


def init_dataframe(expect_index=None):
    # importiong the modules
    # creating the Numpy array
    array = np.array([])

    # creating a list of column names
    column_values = ["1 hour", "12 hours", "24 hours", "36 hours", "48 hours", "60 hour", "72 hour", "84 hours"]
    if expect_index is not None:
        column_values = []
        for i in expect_index:
            column_values.append(f"{i} hours")
            # column_values.append(f"{i} hours")

    # creating a list of index names
    index_values = []

    # creating the dataframe
    df = pd.DataFrame(data=[],
                      index=index_values,
                      columns=column_values)

    # displaying the dataframe
    print(df)
    return df


def get_index_from_dict(method_error, list_index):
    idx = []
    for id, val in enumerate(method_error[1]):
        if val in list_index:
            idx.append(id)
    return idx


def plot_data_errors_on_a_dataset(num_type_display, dataset_order, dict_method_params):
    """
    :param num_type_display: 0:"MSE" or 1:"MAE" or 2:"time order like 1,2,3,..25"
    :param dataset_order: {0,1,2}
    :param dict_method_params: {'auto-tcn': ([0.0085,...], [0.0075,...], [1,2,...]); 'lstm':([0.0095,...], [0.0084,...], [1,2,...])}
    :return:
    """
    expect_index = [1, 12, 24, 36, 48, 60, 72, 84]
    # expect_index = [1, 12, 24, 36, 48, 54, 56, 58, 60, 62, 66, 70, 72, 78, 80, 84]

    df = init_dataframe(expect_index)

    for name_method, m in zip(dict_method_params, markers):
        print(name_method)
        data_plot_param = dict_method_params[name_method][0]
        # data_plot_mae = dict_method_params[name_method][1]
        try:
            # keep the list of hours like 1,2,3,..24, 32,48
            hours_order = dict_method_params[name_method][1]
        except Exception:
            hours_order = list(range(1, len(dict_method_params[name_method][num_type_display]) + 1))

        list_hour = get_index_from_dict(dict_method_params[name_method], expect_index)
        id_index = [hours_order[index] for index in list_hour]
        list_param = [data_plot_param[index] for index in list_hour]
        # error_mae = [data_plot_mae[index] for index in list_hour]

        record = []
        for i, np_param in zip(id_index, list_param):
            record.append(np_param)

        df.loc[name_method] = record

    print(df)
    df.to_csv(f"{list_dataset[dataset_order]}_param.csv")


def compare_delayNet_result():
    dict_method_error = dict()
    # Fedot
    # import numpy as np
    # fedot_errs = np.loadtxt("automl_searching/fedot_err.txt", dtype=float)
    # dict_method_error["fedot"] = fedot_errs.transpose()

    # DelayedNet
    # path_folder = f'auto_correlation/cnu_result/T100_kernal128/*.txt'
    path_folder = f'auto_correlation/cnu_result/T100_kernal32/*.txt'
    listdir = glob.glob(path_folder)
    # print(listdir)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    delay_errs = list_error(listdir)
    dict_method_error["stride_dilated_net"] = delay_errs

    path_folder1 = f"automl_searching/{list_dataset[2]}_result_lstm/*.txt"
    listdir = glob.glob(path_folder1)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    lstm_errs = list_error(listdir)
    dict_method_error["lstm"] = lstm_errs

    # TCN auto-generated search
    path_folder2 = f"automl_searching/{list_dataset[2]}_result_auto/*.txt"
    listdir = glob.glob(path_folder2)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    our_errs = list_error(listdir)
    dict_method_error["auto-tcn"] = our_errs

    #  GRU
    path_folder3 = f"automl_searching/{list_dataset[2]}_result_gru/*.txt"
    listdir = glob.glob(path_folder3)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    gru_errs = list_error(listdir)
    dict_method_error["gru"] = gru_errs

    # AUTO-CORRELATION
    cnu_auto_pth = "auto_correlation/cnu_auto/*.txt"
    listdir = glob.glob(cnu_auto_pth)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    correlation_errs = list_error(listdir)
    dict_method_error["auto-stride_dilated_net"] = correlation_errs

    plot_data_errors_on_a_dataset(0, 2, dict_method_error)


def get_error(path_folder):
    listdir = glob.glob(path_folder)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    return list_error(listdir)


def compare_diff_input_len():
    # list_dataset = ['household', 'spain', 'cnu']
    num_data = 0  # number of dataset observation
    dict_method_error = dict()

    # AUTO-CORRELATION 168 input
    correlation_auto_pth = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_result/100input/*.txt"
    dict_method_error["100input"] = get_error(correlation_auto_pth)
    # AUTO-CORRELATION 100 input
    correlation_auto_pth = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_result/168input/*.txt"
    dict_method_error["168input"] = get_error(correlation_auto_pth)

    plot_data_errors_on_a_dataset(0, num_data, dict_method_error)


def compare_auto_correlation_SPAIN():
    # list_dataset = ['household', 'spain', 'cnu']
    num_data = 1  # number of dataset observation
    dict_method_error = dict()

    # TCN auto-generated search
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_auto/*.txt"
    dict_method_error["auto-tcn"] = get_num_param(path_folder2)

    # LSTM
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_lstm/*.txt"
    dict_method_error["lstm"] = get_num_param(path_folder2)

    # GRU
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_gru/*.txt"
    dict_method_error["gru"] = get_num_param(path_folder2)

    # Autocorrelation-Dilated TCN
    path_folder2 = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_result/*.txt"
    dict_method_error["Autocorrelation-Dilated TCN"] = get_num_param(path_folder2)

    for fn in [2, 3, 4]:
        # AUTO-CORRELATION
        correlation_auto_pth = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_auto_fix_{fn}layers/*.txt"
        dict_method_error[f"auto-stride-{fn}layers"] = get_num_param(correlation_auto_pth)

    plot_data_errors_on_a_dataset(0, num_data, dict_method_error)


def compare_auto_correlation_Household():
    # list_dataset = ['household', 'spain', 'cnu']
    num_data = 0  # number of dataset observation
    dict_method_error = dict()

    # TCN auto-generated search
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_auto/*.txt"
    dict_method_error["auto-tcn"] = get_num_param(path_folder2)

    # LSTM
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_lstm/*.txt"
    dict_method_error["lstm"] = get_num_param(path_folder2)

    # GRU
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_gru/*.txt"
    dict_method_error["gru"] = get_num_param(path_folder2)

    # Autocorrelation-Dilated TCN
    path_folder2 = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_result/*.txt"
    dict_method_error["Autocorrelation-Dilated TCN"] = get_num_param(path_folder2)

    # # AUTO-CORRELATION
    # correlation_auto_pth = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_auto_multi_max2layers/*.txt"
    # dict_method_error["auto-stride-max2layers"] = get_error(correlation_auto_pth)

    for fn in [2, 3, 4]:
        # AUTO-CORRELATION
        correlation_auto_pth = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_auto_fix_{fn}layers/*.txt"
        dict_method_error[f"auto-stride-{fn}layers"] = get_num_param(correlation_auto_pth)

    plot_data_errors_on_a_dataset(0, num_data, dict_method_error)


def list_params_num(listdir):
    listdir = os_sorted(listdir)
    list_params_value = []
    list_order = []
    for filename in listdir:
        params_value = export_keywords_value_from_txt(filename)
        list_params_value.append(params_value)
        # extend to get detail order
        try:
            hour = int(filename[-6:-4])
            list_order.append(hour)
        except Exception:
            try:
                hour = int(filename[-5:-4])
                list_order.append(hour)
            except Exception:
                pass
    return list_params_value, list_order


def get_num_param(path_folder):
    listdir = glob.glob(path_folder)
    listdir.sort(key=lambda x: os.path.getmtime(x))
    return list_params_num(listdir)


def compare_auto_correlation_CNU():
    # list_dataset = ['household', 'spain', 'cnu']
    num_data = 2  # number of dataset observation
    dict_method_params = dict()

    # TCN auto-generated search
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_auto/*.txt"
    dict_method_params["auto-tcn"] = get_num_param(path_folder2)

    # LSTM
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_lstm/*.txt"
    dict_method_params["lstm"] = get_num_param(path_folder2)

    # GRU
    path_folder2 = f"automl_searching/{list_dataset[num_data]}_result_gru/*.txt"
    dict_method_params["gru"] = get_num_param(path_folder2)

    # Autocorrelation-Dilated TCN
    path_folder2 = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_result/*.txt"
    dict_method_params["Autocorrelation-Dilated TCN"] = get_num_param(path_folder2)

    for fn in [2, 3, 4]:
        # AUTO-CORRELATION
        correlation_auto_pth = f"auto_correlation/{list_dataset[num_data]}/{list_dataset[num_data]}_auto_fix_{fn}layers/*.txt"
        dict_method_params[f"auto-stride-{fn}layers"] = get_num_param(correlation_auto_pth)

    plot_data_errors_on_a_dataset(0, num_data, dict_method_params)


if __name__ == '__main__':
    # compare_tcn_auto()
    # compare_delayNet_result()
    # compare_auto_correlation_CNU()
    # compare_auto_correlation_SPAIN()
    compare_auto_correlation_Household()
