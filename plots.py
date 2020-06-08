import matplotlib.pyplot as plt
import numpy as np


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
        y = []
        count = 0
        for alg, data in value.items():
            count += 1
            x.append(alg)
            for i, metrics in data.items():
                if i == 'Voice':
                    y.append(float(metrics['RMSE']))            #Temp version. Rework to SDR, SIR
        fig, ax = plt.subplots(figsize=figure_size)
        ax.set_ylabel('RMSE')                                   #Temp version
        ax.set_title("{0} :".format(key))
        ax.set_xticklabels(x)
        ax.bar(np.arange(0, count), y, width=bar_width)
        # name = "".join((key, ".jpeg"))
        plt.savefig("".join((plot_folder, key, ".pdf")))
    print("".join(("Plots saved in \"./", plot_folder, "\"")))
