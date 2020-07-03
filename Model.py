import numpy as np
import pyroomacoustics as pra
import sys
import copy
from typing import Tuple
from setups import speakers_device_idx
from RecorderClass import Recorder
import threading
from Player import play


def mix(s_input: np.ndarray, data_set: dict, sim: dict):
    S = copy.deepcopy(s_input)

    # choose mix type
    mix_type = sim['mix_type']

    if mix_type == 'linear':
        return mix_linear(S, sim['options'])
    elif mix_type == 'convolutive':
        return mix_convolutive(S, sim['options'])
    elif mix_type == 'convolutive_real':
        return mix_record(S, data_set, sim)
    else:
        print("\033[31m {}".format('Error : Simulation is chosen wrong!'))
        sys.exit()


def mix_linear(S: np.ndarray, opts: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    # Adding noise to signals
    if 'sigma2_awgn' in opts.values():
        S += opts['sigma2_awgn'] * np.random.normal(size=S.shape)

    # Create linear mixing matrix
    M = S.shape[0]  # number of microphones
    corr_coef = 0.5  # "correlation" coefficient
    A = (1 - corr_coef) * np.identity(M) + corr_coef * np.ones(M)

    # "filtered" are the "convolved" (in linear case just multiplied with a scalar coefficient)
    #  but not summed signals
    filtered = []
    for s, a in zip(S, A.T):  # iterates over pairs of source signals (s) and columns of A (a)
        filtered.append(np.outer(a, s))
    filtered = np.array(filtered)

    # Mixture is the sum of this bigger array
    mixed = np.sum(filtered, axis=0)
    return filtered, mixed, {'mixing_matrix': A}


def mix_convolutive(S: np.array, opts: dict) -> Tuple[np.ndarray, np.ndarray, dict]:

    for i in range(S.shape[0]):
        S[i].resize((len(S[i]),))

    N = S.shape[0]  # number of sources
    M = N  # number of microphones
    # ToDo: think about how M is set - currently we resort to "determined" case (M = N),
    # ToDo: but in general it might be something else

    # Some parameters from example on https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html
    # The desired reverberation time and dimensions of the room

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(opts['rt60'], opts['room_dim'])

    # Create room
    room = pra.ShoeBox(opts['room_dim'],
                       fs=opts['fs'],
                       materials=pra.Material(e_absorption),
                       max_order=max_order,
                       sigma2_awgn=opts['sigma2_awgn'])

    # Place the mic. array into the room
    if M == 3:
        # 3 micro case
        R = np.c_[
            [3, 2.87, 1],  # microphone 1
            [3, 2.93, 1],  # microphone 2
            [3, 2.99, 1],  # microphone 3
        ]
    elif M == 2:
        # 2 micro case
        R = np.c_[
            [3, 2.87, 1],  # microphone 1
            [3, 2.93, 1]   # microphone 2
        ]
    else:
        print("Error: wrong number of microphones")
        sys.exit()
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    # Place the sources inside the room
    source_locations = [
        [3., 2., 1.8],  # source 1
        [6., 4., 1.8],  # source 2
        [2., 4.5, 1.8], # source 3
    ]

    # At first we add empty sources in order to record each source separately for SDR/SIR computation later
    # (according to https://github.com/LCAV/pyroomacoustics/blob/pypi-release/examples/bss_example.py)
    for sig, loc in zip(S, source_locations):
        room.add_source(loc, signal=np.zeros_like(sig))

    # Make separate recordings
    filtered = []
    for source, s in zip(room.sources, S):
        # Set only one of the signals
        source.signal[:] = s

        # Simulate & record the signal
        room.simulate()
        filtered.append(room.mic_array.signals)

        # Unset that source's signal (for next iterations)
        source.signal[:] = 0
    filtered = np.array(filtered)

    # Now mixed signals is just the sum
    mixed = np.sum(filtered, axis=0)

    return filtered, mixed, {'room_object': room}


def mix_record(X:np.array, data_set: dict, sim: dict):
    '''Play audio data on Speakers and Record via MiniDSP'''

    # 1. Setup the Voliume of produced data and setup the Recorder
    vol_gain = 5000
    idx = speakers_device_idx()
    X = X * vol_gain
    recorder = Recorder(kwargs=({'fs': data_set['fs'],
                                 'chunk_size': sim['chunk_size'],
                                 'audio_duration': data_set['audio_duration'],
                                 'microphones': sim['microphones']}))

    # 2. Play each source separatly and form the Filtered data
    filtered = filtered_real_data(X, recorder, idx)

    # 3. Make threads for each speaker. Play and record all source data in the same time.
    print('\033[96mStarting record the mixing data...\033[0m')
    threads = []
    for i in range(len(idx)):
        threads.append(threading.Thread(target=play, args=(X[i], idx[i])))

    rec = threading.Thread(target=recorder._record)
    rec.start()
    for thread in threads:
        thread.start()
    rec.join()

    return filtered, recorder._data, {'recorder': recorder}



def filtered_real_data(X, recorder, idx):
    '''Play each source separatly for form the Filtered data '''
    #TODO: Иногда из за кривого опредленения длительности,
    #      размеры листов записанных чанков для разных треков могут не совпадать, тогда выкинет ошибку
    #      -> просто перезапустить
    # UPD: С округлением даты записи до 1 числа после запятой ошибки не возникло в 10 тестах
    import time

    filtered_temp = []
    for i in range(X.shape[0]):
        rec = threading.Thread(target=recorder._record)
        try:
            s = threading.Thread(target=play, args=(X[i], idx[i]))
        except IndexError:
            break
        print('\033[96mStarting record the filtered source \033[0m')
        rec.start()
        s.start()
        rec.join()
        time.sleep(2)
        filtered_temp.append(recorder._data)

    filtered = []
    for i in range(len(filtered_temp)):
        filtered.append(np.concatenate(filtered_temp[i], axis=1))

    return np.array(filtered)

