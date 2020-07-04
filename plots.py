import matplotlib.pyplot as plt
import numpy as np
import os


def plot(X: np.array):
    """Plot filtered and audio data"""
    figure_size = (15, 7)
    filt_plot_folder = "plots/filtered/"
    mixed_plot_folder = "plots/mixed/"

    fig, ax = plt.subplots(X.shape[0], X.shape[1], sharex='col', sharey='row', figsize=figure_size)
    plt.grid()
    x = 0
    for i in X:
        y = 0
        for k in range(i.shape[0]):
            ax[y, x].plot(i[y])
            ax[y, x].grid()
            y += 1
        x += 1
    #plt.show()


def print_results(results: dict):
    print("______________________________")
    for key, value in results.items():
        print("{0} :".format(key))
        for alg, data in value.items():
            print("\t{0} :".format(alg))
            for audio_type, metrics in data.items():
                print("\t\t{0}: {1}".format(audio_type, metrics))
    print("______________________________")


def rework_dict(sims: list):
    results = {}
    for sim in sims:
        dic = {sim['name']: {}}
        dic2 = dic[sim['name']]
        for alg in sim['algs']:
            dic2.update({alg['name']: alg['metrics']})
        results.update(dic)
    return results


def plot_metrics(rew_sims: dict, dir_plots: str):
    """Plot bar plots for presentations"""
    # Setups
    figure_size = (10, 7)
    bar_width = 0.5
    metr_type = ['RMSE', 'SDR', 'SIR']

    for sim_name, rew_algs in rew_sims.items():
        x = [0]
        RMSE = []
        SDR = []
        SIR = []
        metr_value = [RMSE, SDR, SIR]
        count = 0
        for alg, data in rew_algs.items():
            count += 1
            x.append(alg)
            for i, metrics in data.items():
                if i == 'Voice':
                    for z in range(len(metr_type)):
                        metr_value[z].append(float(np.round(np.mean(metrics[metr_type[z]]), 2)))

        for k in range(len(metr_type)):
            folder = "{}/{}".format(dir_plots, sim_name)
            try:
                os.mkdir(folder)
            except OSError:
                pass
            fig, ax = plt.subplots(figsize=figure_size)
            ax.set_ylabel(metr_type[k], fontsize=14, fontweight="bold")
            ax.set_title("{0} :".format(sim_name), fontsize=18, fontweight="bold")
            ax.set_xticklabels(x)
            ax.bar(np.arange(0, count), metr_value[k], width=bar_width)
            plt.savefig("{}/{}.pdf".format(folder, metr_type[k]) )
    print("".join(("Plots saved in \"./", dir_plots, "\"")))


def plot_sim_data_set_metrics(sim: dict, data_set: dict, dir_sim_plots: str):
    """Plot bar plots for specific sim for presentations"""
    # Setups
    figure_size = (10, 7)
    bar_width = 0.5
    ds_type = data_set['type']

    x = []
    metrics = {
        'RMSE': [],
        'SDR': [],
        'SIR': []
    }

    for alg in sim['algs']:
        x.append(alg['name'])
        for metric_name, metric_list in metrics.items():
            metric_list.append(np.mean(alg['metrics'][ds_type][metric_name]))

    dir_sim_plots_type = "{}/{}".format(dir_sim_plots, ds_type)
    if not os.path.isdir(dir_sim_plots_type):
        os.mkdir(dir_sim_plots_type)

    for metric_name, metric_list in metrics.items():
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_ylabel(metric_name, fontsize=14, fontweight="bold")
        ax.set_title("Sim: {}; data set: {}".format(sim['name'], ds_type), fontsize=18, fontweight="bold")
        ax.set_xticklabels(x)
        ax.bar(np.arange(0, len(x)), metrics[metric_name], width=bar_width)
        plt.savefig("{}/{}.pdf".format(dir_sim_plots_type, metric_name))
