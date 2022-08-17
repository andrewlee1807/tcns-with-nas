from matplotlib import pyplot as plt


def export_mse_mae_from_txt(pth):
    error_singles = []
    error_multis = []
    with open(pth, 'r') as f:
        data = f.readlines()
    for line in data:
        break_line = line.split(" ")
        error1 = float(break_line[2])
        error2 = float(break_line[5])
        error_singles.append(error1)
        error_multis.append(error2)
    return error_singles, error_multis


def plot_data_error():
    fig, ax = plt.subplots()

    ax.plot(list(range(1, len(error_singles) + 1)), error_singles,
            marker='.', linestyle='-', linewidth=0.5, label='error_singles')

    ax.plot(list(range(1, len(error_singles) + 1)), error_multis,
            marker='o', markersize=8, linestyle='-', label='error_multis')
    ax.set_ylabel("MSE")
    ax.set_xlabel("Days")

    ax.legend()
    plt.show()


# plot_data_error()
error_singles, error_multis = export_mse_mae_from_txt(
    r"\\168.131.153.57\andrew\Time Series\TSDatasets\tcn_analysis\tcn_output_SinglevsMulti.txt")

plot_data_error()
