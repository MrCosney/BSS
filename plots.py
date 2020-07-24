import matplotlib.pyplot as plt
import numpy as np
import os


def plot_original(S: np.array, dir_plots: str, name_prefix: str = "", samples: int = 20000):
    """Plot original audio data"""
    figure_size = (15, 7)

    fig, ax = plt.subplots(S.shape[0], sharex='col', sharey='row', figsize=figure_size)
    fig.suptitle("Original", fontsize=18, fontweight="bold")
    plt.grid()
    for si, s in enumerate(S):
        ax[si].plot(s[:samples])
        ax[si].grid()
        ax[si].set_title("Source {}".format(si+1), fontsize=18, fontweight="bold")
    plt.savefig("{}/{}original.pdf".format(dir_plots, name_prefix))
    # plt.show()


def plot_filtered(filtered: np.array, dir_plots: str, name_prefix: str = "", samples: int = 20000):
    """Plot filtered audio data"""
    figure_size = (15, 7)

    fig, ax = plt.subplots(filtered.shape[0], filtered.shape[1], sharex='col', sharey='row', figsize=figure_size)
    fig.suptitle("Filtered, sources x microphones", fontsize=18, fontweight="bold")

    for fi, f in enumerate(filtered):
        for mi, m in enumerate(f):
            # TODO: Сейчас строит только первые 20к отсчетов для отслеживания задержки
            ax[fi, mi].plot(m[:samples])
            ax[fi, mi].grid()
            ax[fi, mi].set_title("Source {}, microphone {}".format(fi+1, mi+1), fontsize=18, fontweight="bold")

    plt.grid()
    plt.savefig("{}/{}filtered.pdf".format(dir_plots, name_prefix))
    # plt.show()


def plot_mixed(mixed: np.array, dir_plots: str, name_prefix: str = "", samples: int = 20000):
    """Plot filtered and audio data"""
    figure_size = (15, 7)

    fig, ax = plt.subplots(mixed.shape[0], sharex='col', sharey='row', figsize=figure_size)
    fig.suptitle("Mixed", fontsize=18, fontweight="bold")
    plt.grid()
    for mi, m in enumerate(mixed):
        ax[mi].plot(m[:samples])
        ax[mi].grid()
        ax[mi].set_title("Microphone {}".format(mi+1), fontsize=18, fontweight="bold")
    plt.savefig("{}/{}mixed.pdf".format(dir_plots, name_prefix))
    # plt.show()


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
    metr_type = ['SDR', 'SIR', 'SAR', 'P']

    for sim_name, rew_algs in rew_sims.items():
        x = [0]
        SDR = []
        SIR = []
        SAR = []
        P = []
        metr_value = [SDR, SIR, SAR, P]
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
            plt.savefig("{}/{}.pdf".format(folder, metr_type[k]))
    print("".join(("Plots saved in \"./", dir_plots, "\"")))


def plot_sim_data_set_metrics(sim: dict, data_set: dict, dir_sim_plots: str):
    """Plot bar plots for specific sim for presentations"""
    # Setups
    figure_size = (10, 7)
    bar_width = 0.5
    ds_type = data_set['type']

    x = []
    metrics = {
        'SDR': [],
        'SIR': [],
        'SAR': [],
        'P': []
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
        ax.bar(np.arange(len(x)), metrics[metric_name], width=bar_width)
        plt.xticks(np.arange(len(x)), x, rotation=45)
        plt.savefig("{}/{}.pdf".format(dir_sim_plots_type, metric_name))

def plot_boxes(SDR, SAR, SIR, sim_name, dir_sim_box_plots):
    """Plot box-plots for all sim for presentations"""

    labels=  ['AUXIVA_5', 'AUXIVA_10', 'AUXIVA_50', 'ILRMA', 'AIRES (rtap-opt)', 'AIRES (batch)']
    names = {'SDR': SDR, 'SAR': SAR, 'SIR': SIR}
    for name, value in names.items():
        var = []
        for i in range(len(value[0])):
            z = []
            for alg in value:
                if np.isnan(alg[i]) == True:
                    print("yep")
                    continue
                z.append(alg[i])
            var.append(z)
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 6)
        #plt.figure(figsize=(12, 8))
        ax.set_title(sim_name, fontsize=16, fontweight="bold")
        ax.set_ylabel("".join((name, ",[dB]")), fontsize=12, fontweight="bold")
        ax.boxplot(var, labels=labels[:len(SDR[0])])
        plt.savefig("{}/{}.pdf".format(dir_sim_box_plots, name))




