import pyroomacoustics as pra
import matplotlib.pyplot as plt
from Player import *


def makeroom(fs, data: np.array, opt: dict):
    for i in range(data.shape[0]):
        data[i].resize((len(data[i]),))
    room = pra.ShoeBox([7, 5, 3.2], fs=fs, absorption=opt['absorp'], max_order=opt['orders'], sigma2_awgn=opt['awgn'])
    '''Place the sources inside the room'''
    if data.shape[0] == 3:
        room.add_source([3., 2., 1.8], signal=data[0])
        room.add_source([6., 4., 1.8], signal=data[1])
        room.add_source([2., 4.5, 1.8], signal=data[2])
        R = np.c_[
            [3, 2.87, 1],  # microphone 1
            [3, 2.93, 1],  # microphone 2
            [3, 2.99, 1],  # microphone 3
        ]
    else:
        room.add_source([3., 2., 1.8], signal=data[0])
        room.add_source([6., 4., 1.8], signal=data[1])
        R = np.c_[
            [3, 2.87, 1],  # microphone 1
            [3, 2.93, 1]  # microphone 2
        ]
    #delays = [1., 0.]   #TODO: Check for delay
    '''Place the mic. array into the room'''
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    room.compute_rir()
    room.simulate()
    return room.mic_array.signals, room
