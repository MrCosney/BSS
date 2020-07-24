import numpy as np
import pyroomacoustics as pra
import sys
import copy
from typing import Tuple
from RecorderClass import Recorder
import threading
from Player import play


def mix(s_input: np.ndarray, sim: dict, data_set: dict):
    S = copy.deepcopy(s_input)

    # choose mix type
    mix_type = sim['mix_type']

    if mix_type == 'linear':
        [filtered, mixed, mao] = mix_linear(S, sim)
    elif mix_type == 'convolutive':
        [filtered, mixed, mao] = mix_convolutive(S, sim, data_set)
    elif mix_type == 'experimental':
        [filtered, mixed, mao] = mix_experimental(S, sim, data_set)
    else:
        print("\033[31m {}".format('Error: Simulation is chosen wrong!'))
        sys.exit()

    sim['mix_additional_outputs'] = mao
    return filtered, mixed, sim


def mix_linear(S: np.ndarray, sim: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    # Get parameters
    opts = sim['options']
    N = S.shape[0]  # number of sources
    M = sim['microphones'] if 'microphones' in sim else N  # number of microphones

    # Adding noise to signals
    if 'sigma2_awgn' in opts.values():
        S += opts['sigma2_awgn'] * np.random.normal(size=S.shape)

    # Create linear mixing matrix
    corr_coef = 0.5  # "correlation" coefficient
    A = (1 - corr_coef) * np.eye(M, N) + corr_coef * np.ones(M, N)

    # "filtered" are the "convolved" (in linear case just multiplied with a scalar coefficient)
    #  but not summed signals
    filtered = []
    for s, a in zip(S, A.T):  # iterates over pairs of source signals (s) and columns of A (a)
        filtered.append(np.outer(a, s))
    filtered = np.array(filtered)

    # Mixture is the sum of this bigger array
    mixed = np.sum(filtered, axis=0)
    return filtered, mixed, {'mixing_matrix': A}


def hexagonal_points(d: float) -> np.ndarray:
    return d * np.array([[-1, 0, 0],
                         [-1 / 2, 3 ** 0.5 / 2, 0],
                         [-1 / 2, -3 ** 0.5 / 2, 0],
                         [0, 0, 0],
                         [1 / 2, 3 ** 0.5 / 2, 0],
                         [1 / 2, -3 ** 0.5 / 2, 0],
                         [1, 0, 0]]).T


def mix_convolutive(S: np.array, sim: dict, data_set: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    # Get parameters
    import matplotlib.pyplot as plt
    opts = sim['options']
    N = S.shape[0]  # number of sources
    M = sim['microphones'] if 'microphones' in sim else N  # number of microphones

    # Some parameters from example on https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html
    # The desired reverberation time and dimensions of the room

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(opts['rt60'], opts['room_dim'])

    # Create room
    room = pra.ShoeBox(opts['room_dim'],
                       fs=data_set['fs'],
                       materials=pra.Material(e_absorption),
                       max_order=max_order,
                       sigma2_awgn=opts['sigma2_awgn'])
    # Microphone locations for hexagonal array
    array_loc = np.array([[3], [2], [0.5]])
    micro_locs = hexagonal_points(sim['microphones_distance'])
    micro_locs += array_loc

    # Check that required number of microphones has it's locations
    if micro_locs.shape[0] < M:
        raise ValueError('{} microphones required, but only {} microphone locations specified'
                         .format(M, micro_locs.shape[0]))

    # Select as much microphones as needed
    R = micro_locs[:, :M]

    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    #room.add_microphone_array(pra.Beamformer(R, room.fs))
    # Place the sources inside the room
    source_locs = np.array([
        [3., 3, 0.85],   # source 1
        [3., 1, 0.85],   # source 2
        [2., 4.5, 1.8],  # source 3
    ])

    # Check that required number of microphones has it's locations
    if source_locs.shape[0] < N:
        raise ValueError('{} sources required, but only {} source locations specified'
                         .format(N, source_locs.shape[0]))

    # At first we add empty sources in order to record each source separately for SDR/SIR computation later
    # (according to https://github.com/LCAV/pyroomacoustics/blob/pypi-release/examples/bss_example.py)
    for sig, loc in zip(S, source_locs):
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
    #room.plot(freq=[1000, 2000], img_order=0)
    #plt.show()
    return filtered, mixed, {'room_object': room}


def mix_experimental(S: np.array, sim: dict, data_set: dict) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Play audio data on Speakers and Record via MiniDSP"""

    # Get parameters
    opts = sim['options']
    N = S.shape[0]  # number of sources
    M = sim['microphones'] if 'microphones' in sim else N  # number of microphones

    # 1. Apply volume gain
    S = S * opts['volume_gain']

    # 2. Get available speakers
    idxs = speakers_device_idx()
    if len(idxs) < N:
        raise ValueError('{} sources required, but only {} speakers available'
                         .format(N, len(idxs)))

    # 3. Setup the Recorder
    recorder = Recorder(fs=data_set['fs'],
                        chunk_size=sim['chunk_size'],
                        audio_duration=data_set['audio_duration'],
                        microphones=M)

    # 4. Play each source separately and form the 'filtered' data
    filtered = record_filtered(S, recorder, idxs)

    # 5. Play and record all source data at the same time.
    mixed = record_mixed(S, recorder, idxs)

    return filtered, mixed, {'recorder': recorder}


def speakers_device_idx():
    """This functions get device index for output source data to speakers"""
    import pyaudio

    p = pyaudio.PyAudio()
    device_idx = []
    print('\033[96mList of Speakers:\033[0m')
    for i in range(p.get_device_count()):
        if "TF-PS1234B Stereo" in p.get_device_info_by_index(i)['name']:
            print("\t", p.get_device_info_by_index(i)['name'])
            device_idx.append(i)

    if len(device_idx) == 0:
        print("\033[31m {}" .format('Error : No Speakers!'))
        sys.exit()

    return device_idx


def record_filtered(S: np.ndarray, recorder: Recorder, idxs: list) -> np.ndarray:
    """Play each source separately for form the Filtered data"""
    filtered = []

    for idx, s in zip(idxs, S):
        rThread = threading.Thread(target=recorder.record)
        sThread = threading.Thread(target=play, args=(s, idx))
        print('\033[96mStart recording the filtered data...\033[0m')
        rThread.start()
        sThread.start()
        rThread.join()
        sThread.join()
        filtered.append(recorder.get_data())
    t_fist = [filtered[0][:100000], filtered[1][:100000]]
    filtered = np.array(filtered)

    return np.array(filtered)


def record_mixed(S: np.ndarray, recorder: Recorder, idxs: list) -> np.ndarray:
    print('\033[96mStart recording the mixed data...\033[0m')

    # Create speaker threads
    sThreads = []
    for idx, s in zip(idxs, S):
        sThreads.append(threading.Thread(target=play, args=(s, idx)))

    rThread = threading.Thread(target=recorder.record)
    rThread.start()
    for sThread in sThreads:
        sThread.start()
    rThread.join()
    for sThread in sThreads:
        sThread.join()

    return recorder.get_data()

