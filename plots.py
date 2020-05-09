import matplotlib.pyplot as plt

def plots(orig1, orig2, orig3, mixed, unmixed, rmse, name, th=1 ,th1=1.7):
    plt.style.use('seaborn')
    fig, axes = plt.subplots(figsize=(16, 5), nrows=3, ncols=3)
    axes[0][0].set_title('Original Signals')
    axes[0][0].plot(orig1, color='black', linewidth=th1)
    axes[1][0].plot(orig2, color='black', linewidth=th1)
    axes[2][0].plot(orig3, color='black', linewidth=th1)

    axes[0][1].set_title('Mixed Signals')
    axes[0][1].plot(mixed[0], color='black', linewidth=th)
    axes[0][1].plot(orig1, alpha=0.4, color='red', linewidth=th1)

    axes[1][1].plot(mixed[1], color='black', linewidth=th)
    axes[1][1].plot(orig2, alpha=0.4, color='red', linewidth=th1)

    axes[2][1].plot(mixed[2], color='black', linewidth=th)
    axes[2][1].plot(orig3, alpha=0.4, color='red', linewidth=th1)

    axes[0][2].set_title('Reconstructed Signals')
    axes[0][2].plot((unmixed[0]), color='black', linewidth=th)
    axes[1][2].plot((unmixed[1]), color='black', linewidth=th)
    axes[2][2].plot((unmixed[2]), color='black', linewidth=th)

    axes[0][2].set_ylabel('RMSE = ' + str(round(rmse[0], 3)))
    axes[0][2].yaxis.set_label_position("right")
    axes[1][2].set_ylabel('RMSE = ' + str(round(rmse[1], 3)))
    axes[1][2].yaxis.set_label_position("right")

    axes[2][2].set_ylabel('RMSE = ' + str(round(rmse[2], 3)))
    axes[2][2].yaxis.set_label_position("right")
    plt.savefig(name)
    #plt.show()

