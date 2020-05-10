import matplotlib.pyplot as plt
import numpy as np

def plots(orig, mixed, unmixed, rmse, name, th=1 ,th1=1.7):
    plt.style.use('seaborn')
    fig, axes = plt.subplots(figsize=(16, 5), nrows=orig.shape[0], ncols=3)
    '''Orig plot'''
    axes[0][0].set_title('Original Signals')
    for i in range(orig.shape[0]):
        axes[i][0].plot(orig[i], color='black', linewidth=th1)
    '''Mix plot'''
    axes[0][1].set_title('Mixed Signals')
    for i in range(mixed.shape[0]):
        axes[i][1].plot(mixed[i], color='black', linewidth=th)
        axes[i][1].plot(orig[i], alpha=0.4, color='red', linewidth=th1)
    '''Unmix plot'''
    axes[0][2].set_title('Reconstructed Signals')
    for i in range(unmixed.shape[0]):
        axes[i][2].plot((unmixed[i]), color='black', linewidth=th)

    axes[0][2].set_ylabel('RMSE = ' + str(round(rmse[0], 3)))
    axes[0][2].yaxis.set_label_position("right")
    axes[1][2].set_ylabel('RMSE = ' + str(round(rmse[1], 3)))
    axes[1][2].yaxis.set_label_position("right")
    z = unmixed.shape[0]
    if unmixed.shape[0] > 2:
        axes[2][2].set_ylabel('RMSE = ' + str(round(rmse[2], 3)))
        axes[2][2].yaxis.set_label_position("right")

    plt.savefig(name)
    #plt.show()

def swap_lines(object,fr, on):
    '''Swap n and m lines . For some algorithms'''
    temp = np.copy(object)
    object[fr] = temp[on]
    object[on] = temp[fr]
    return object

def swap(JU, JU_s, icaa_audio, icaa):
    """Swapping line after unmixing for correct display on plots"""
    swap_lines(JU, 0, 1)
    swap_lines(JU_s, 0, 1)
    swap_lines(icaa_audio, 0, 2)
    swap_lines(icaa_audio, 0, 1)
    '''Mirroring '''
    JU[1] *= -1
    JU[2] *= -1
    JU_s[1] *= -1
    JU_s[2] *= -1
    icaa_audio[2] *= -1
    icaa[2] *= -1
