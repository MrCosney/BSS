import pyroomacoustics as pra
import matplotlib.pyplot as plt
from Player import *

def makeroom(fs, sig1, sig2, sig3):
    """Place sources at the position in the room.
    Size of the room [x,y,z]
    Absorption of the wall (0 - 1)
    Max_order - number of reflection of the wall
    """
    room = pra.ShoeBox([7, 5, 3.2], fs=fs, absorption=0.35, max_order=0)
    '''Place the sources inside the room'''
    room.add_source([3., 2., 1.8], signal=sig1)
    room.add_source([6., 4., 1.8], signal=sig2)
    room.add_source([2., 4.5, 1.8], signal=sig3)
    '''Place the mic. array into the room'''
    R = np.c_[
        [3, 2.87, 1],  # microphone 1
        [3, 2.93, 1],  # microphone 2
        [3, 2.99, 1],  # microphone 3
    ]
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    '''Plot the sim room'''
    fig, ax = room.plot()
    ax.set_xlim([-1, 8])
    ax.set_ylim([-1, 7])
    ax.set_zlim([0, 3])
    plt.show()
    '''Plot the sim room with reflections'''
    room.image_source_model(use_libroom=True)
    fig, ax = room.plot(img_order=3)
    fig.set_size_inches(18.5, 10.5)
    plt.show()
    '''Plot Impulse Response of the Room'''
    room.plot_rir()
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.show()

    room.simulate()
    #play(room.mic_array.signals[0, :] * 5000)
    #play(room.mic_array.signals[1, :] * 5000)
    #play(room.mic_array.signals[2, :] * 5000)
    return room.mic_array.signals
