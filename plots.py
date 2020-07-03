import matplotlib.pyplot as plt
import numpy as np
import os


def plot(X: np.array):
    '''Plot filtered and audio data'''
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



def rework_dict(d_data: list):
    results = {}
    for sim in d_data:
        dic = {sim['name']: {}}
        dic2 = dic[sim['name']]
        for alg in sim['algs']:
            dic2.update({alg['name']: alg['metrics']})
        results.update(dic)
    print("______________________________")
    for key, value in results.items():
        print("{0} :".format(key))
        for alg, data in value.items():
            print("\t{0} :".format(alg))
            for audio_type, metrics in data.items():
                print("\t\t{0}: {1}".format(audio_type, metrics))
    print("______________________________")
    return results


def plot_metrics(d_data: dict):
    '''Plot bar plots for presentations'''
    # Setups
    figure_size = (10, 7)
    plot_folder = "plots/"
    bar_width = 0.5

    for key, value in d_data.items():
        x = [0]
        RMSE = []
        SDR = []
        SIR = []
        metr_type = ['RMSE', 'SDR', 'SIR']
        metr_value = [RMSE, SDR, SIR]
        count = 0
        for alg, data in value.items():
            count += 1
            x.append(alg)
            for i, metrics in data.items():
                if i == 'Voice':
                    for z in range(len(metr_type)):
                        metr_value[z].append(float(np.round(np.mean(metrics[metr_type[z]]), 2)))

        for k in range(len(metr_type)):
            folder = "".join((plot_folder, key, "/"))
            try:
                os.mkdir(folder)
            except OSError:
                pass
            fig, ax = plt.subplots(figsize=figure_size)
            ax.set_ylabel(metr_type[k], fontsize=14, fontweight="bold")
            ax.set_title("{0} :".format(key), fontsize=18, fontweight="bold")
            ax.set_xticklabels(x)
            ax.bar(np.arange(0, count), metr_value[k], width=bar_width)
            plt.savefig("".join((folder, metr_type[k], ".pdf")))
    print("".join(("Plots saved in \"./", plot_folder, "\"")))
