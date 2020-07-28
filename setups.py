from algs.Algorithms import *
from itertools import combinations
from pathlib import Path


def setups():
    wav_folder = "Audio/Original"
    files = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav", "7.wav", "8.wav", "9.wav", "10.wav"]
    # files = ["1.wav", "2.wav"]
    # files = ["Man.wav", "Woman.wav", "Announcer.wav"]
    data_sets = [
        {
            'name': "".join([Path(f).stem for f in fs]),
            'fs': 16000,
            'data': ['{}/{}'.format(wav_folder, f) for f in fs],
            'file_names': [Path(f).stem for f in fs]
        } for fs in combinations(files, 2)
    ]

    # Offline algorithms - those that are used for offline (batch) simulations
    algs_batch = [
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
     {'name': 'AIRES (offline)',  'func': AIRES_new_offline, 'state': {}, 'options': {}}
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
        {'name': 'AIRES (online)', 'func': AIRES_new_online, 'state': {}, 'options': {}}
    ]

    # Environment (room) options for convolutive simulations
    env_options = {
        'rt60': 0.9,
        'room_dim': [6.4, 3.7, 3.4],
        'sigma2_awgn': 0,
        'volume_gain': 5000
    }

    # Distance between microphones [m]
    mic_dist = 0.05

    # Convolutive, batch, for 2 microphones
    sim_batch_2 = {
        'name': 'Simulated_Batch_M2_S2',
        'mix_type': 'convolutive',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'batch',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': mic_dist,
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
        'microphones_distance': mic_dist,
        'data_sets': data_sets,
        'chunk_size': 2048,
        'env_options': env_options,
        'algs': algs_onln
    }

    expt_batch_2 = {
        'name': 'Experimental_Batch_M2_S2',
        'mix_type': 'experimental',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'batch',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': mic_dist,
        'data_sets': data_sets,
        'chunk_size': 2048,
        'env_options': env_options,
        'algs': algs_batch
    }

    expt_online_2 = {
        'name': 'Experimental_Online_M2_S2',
        'mix_type': 'experimental',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'online',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': mic_dist,
        'data_sets': data_sets,
        'chunk_size': 2048,
        'env_options': env_options,
        'algs': algs_onln
    }

    sims = [
        sim_batch_2,
        sim_online_2,
        # expt_batch_2,
        # expt_online_2
    ]
    return sims, data_sets
