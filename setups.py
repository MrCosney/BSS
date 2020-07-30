from algs.Algorithms import *
from itertools import combinations
from pathlib import Path


def setups():
    wav_folder = "Audio/Original"
    # files = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav", "7.wav", "8.wav", "9.wav", "10.wav"]
    files = ["1.wav", "2.wav"]
    # files = ["Man.wav", "Woman.wav", "Announcer.wav"]
    data_sets = [
        {
            'name': "".join([Path(f).stem for f in fs]),
            'fs': 16000,
            'data': ['{}/{}'.format(wav_folder, f) for f in fs],
            'file_names': [Path(f).stem for f in fs]
        } for fs in combinations(files, 2)
    ]

    # So far Beamformer_Perceptual looks like the best among working beamformers, although all of them have very low
    # SAR but acceptable SIR. SDR is low due to low SAR. The separation is audible in recordings. All beamformers work
    # only for 2 sources in simulated by pyroom acoustics environment.
    # Available (working) beamformers from pyroomacoustics:
    # - Beamformer_Max_UDR, Beamformer_Delay_And_Sum, Beamformer_Perceptual (best), and Beamformer_MVDR (second best)

    # Offline algorithms - those that are used for offline (batch) simulations
    algs_batch = [
     {'name': 'Beamform (Perceptual)', 'func': Beamformer_Perceptual, 'state': {}, 'options': {'stft_size': 1024,
                                                                                               'nPaths': 1}},
     {'name': 'ILRMA (MATLAB)',   'func': ILRMA_MATLAB,  'state': {}, 'options': {'stft_size': 2048,
                                                                                  'iter': 50,
                                                                                  'nBases': 10}},
     {'name': 'ILRMA (Pyroom)',   'func': ILRMA,         'state': {}, 'options': {'stft_size': 256,
                                                                                  'iter': 50,
                                                                                  'nBases': 10}},
     {'name': 'AUXIVA (MATLAB)',  'func': AuxIVA_MATLAB, 'state': {}, 'options': {'stft_size': 512,
                                                                                  'iter': 100}},
     {'name': 'AUXIVA (Pyroom)',  'func': AuxIVA,        'state': {}, 'options': {'stft_size': 256,
                                                                                  'iter': 100}},
     {'name': 'AIRES (offline)',  'func': AIRES_new_offline, 'state': {}, 'options': {'max_delay': 20,
                                                                                      'iter': 30}}
    ]

    # Online algorithms - those that are used for online (chunk-by-chunk) simulations
    algs_onln = [
        {'name': 'ILRMA (MATLAB)',   'func': ILRMA_MATLAB,  'state': {}, 'options': {'stft_size': 256,
                                                                                     'iter': 10,
                                                                                     'nBases': 2}},
        {'name': 'ILRMA (Pyroom)',   'func': ILRMA,         'state': {}, 'options': {'stft_size': 256,
                                                                                     'iter': 10,
                                                                                     'nBases': 2}},
        {'name': 'AUXIVA (MATLAB)',  'func': AuxIVA_MATLAB, 'state': {}, 'options': {'stft_size': 256,
                                                                                     'iter': 10}},
        {'name': 'AUXIVA (Pyroom)',  'func': AuxIVA,        'state': {}, 'options': {'stft_size': 256,
                                                                                     'iter': 10}},
        {'name': 'AIRES (online)', 'func': AIRES_new_online, 'state': {}, 'options': {'max_delay': 20,
                                                                                      'iter_p_block': 2,
                                                                                      'blocks_memory': 1}}
    ]

    # Environment (room) options for convolutive simulations

    # Location of the center of the hexagonal microphone array
    array_location = np.array([[3], [2], [0.5]])

    # Distance between microphones [m]
    mic_dist = 0.05

    # Locations of the sources
    source_locs = np.array([
        [3., 3, 0.85],   # source 1
        [3., 1, 0.85],   # source 2
        [5., 2, 0.85],  # source 3
        [4., 3, 1.85],  # source 3
    ])

    env_options = {
        'rt60': 0.75,
        'room_dim': [6.4, 3.7, 3.4],
        'sigma2_awgn': 5e-7,
        'volume_gain': 5000,
        'microphones_distance': mic_dist,
        'micro_locations': array_location + hexagonal_points(mic_dist),
        'source_locations': source_locs
    }

    # Chunk size for online simulations
    chunk_size = 2048

    # Convolutive, batch, for 2 microphones
    sim_batch_2 = {
        'name': 'Simulated_Batch_M2_S2',
        'mix_type': 'convolutive',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'batch',  # 'batch', 'online'
        'sources': 2,
        'microphones': 7,
        'data_sets': data_sets,
        'env_options': env_options,
        'algs': algs_batch
    }

    # Convolutive, online, for 2 microphones
    sim_online_2 = {
        'name': 'Simulated_Online_M2_S2',
        'mix_type': 'convolutive',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'online',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'data_sets': data_sets,
        'chunk_size': chunk_size,
        'env_options': env_options,
        'algs': algs_onln
    }

    expt_batch_2 = {
        'name': 'Experimental_Batch_M2_S2',
        'mix_type': 'experimental',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'batch',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'data_sets': data_sets,
        'chunk_size': chunk_size,
        'env_options': env_options,
        'algs': algs_batch
    }

    expt_online_2 = {
        'name': 'Experimental_Online_M2_S2',
        'mix_type': 'experimental',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'online',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'data_sets': data_sets,
        'chunk_size': chunk_size,
        'env_options': env_options,
        'algs': algs_onln
    }

    sims = [
        sim_batch_2,
        # # sim_online_2,
        # expt_batch_2,
        # expt_online_2
    ]
    return sims, data_sets


def hexagonal_points(d: float) -> np.ndarray:
    return d * np.array([[-1, 0, 0],
                         [-1 / 2, 3 ** 0.5 / 2, 0],
                         [-1 / 2, -3 ** 0.5 / 2, 0],
                         [0, 0, 0],
                         [1 / 2, 3 ** 0.5 / 2, 0],
                         [1 / 2, -3 ** 0.5 / 2, 0],
                         [1, 0, 0]]).T
