import matplotlib.pyplot as plt

def plots2(orig1, orig2, mixed, unmixed,name, th=1 ,th1=1.7):
    plt.style.use('seaborn')
    fig, axes = plt.subplots(figsize=(16, 5), nrows=2, ncols=3)
    axes[0][0].set_title('Original Signals')
    axes[0][0].plot(orig1, color='black', linewidth=th1)
    axes[1][0].plot(orig2, color='black', linewidth=th1)

    axes[0][1].set_title('Mixed Signals')
    axes[0][1].plot(mixed[0], color='black', linewidth=th)
    axes[0][1].plot(orig1, alpha=0.4, color='red', linewidth=th1)

    axes[1][1].plot(mixed[1], color='black', linewidth=th)
    axes[1][1].plot(orig2, alpha=0.4, color='red', linewidth=th1)

    axes[0][2].set_title('Reconstructed Signals')
    axes[0][2].plot((unmixed[0]), color='black', linewidth=th)
    axes[1][2].plot((unmixed[1]), color='black', linewidth=th)
    plt.savefig(name)
    #plt.show()

