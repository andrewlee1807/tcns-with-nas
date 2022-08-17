"""        kernel_size 7,  and
        nb_filters: 16,
        dropout_rate: 0.0
        layer_stride1: 9
        layer_stride2: 3"""

import glob
import os

keywords = "Showing"
list_dataset = ['household', 'spain', 'cnu']
num_dataset_observation = 2

list_hyper_parameters = ["kernel_size", "nb_filters", "dropout_rate", "nb_stride_block", "layer_stride1",
                         "layer_stride2", "layer_stride3"]


def export_keywords_value_from_txt(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()

    find_params_line = lambda x: keywords in lines[x]
    end_index = list(filter(find_params_line, range(len(lines) - 1, -1, -1)))[0]
    start_index = end_index + 4
    end_index = start_index + 6
    hyper_parameters_generated = dict()
    for i, name in zip(range(start_index, end_index + 1), list_hyper_parameters):
        index_str = lines[i].find(name)
        hyper_para_value = lines[i].split(" ")[index_str + 1]
        hyper_parameters_generated[name] = hyper_para_value

    return hyper_parameters_generated


def list_params_num(listdir):
    for i, filename in enumerate(listdir):
        params_value = export_keywords_value_from_txt(filename)
        print(i + 1, params_value)


print(list_dataset[num_dataset_observation])
correlation_auto_pth = f"auto_correlation/{list_dataset[num_dataset_observation]}_auto_multi_layers/*.txt"
listdir = glob.glob(correlation_auto_pth)
listdir.sort(key=lambda x: os.path.getmtime(x))
list_params_num(listdir)
