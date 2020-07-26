from algs.Algorithms import *
from itertools import combinations
from pathlib import Path


def setups():
    SDRR = []
    SARR = []
    SIRR = []
    np.random.seed(0)
    n_samples = 20000
    time = np.linspace(0, 8, n_samples)
    wav_folder = "Audio/Original"
    files = ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav"] #"6.wav", "7.wav", "8.wav", "9.wav", "10.wav"]
    #files = ["1.wav", "2.wav", "3.wav"]
    # files = ["Man.wav", "Woman.wav", "Announcer.wav"]
    data_sets = [
        {
            'name': "".join([Path(f).stem for f in fs]),
            'fs': 16000,
            'data': ['{}/{}'.format(wav_folder, f) for f in fs],
            'file_names': [Path(f).stem for f in fs]
        } for fs in combinations(files, 2)
    ]
    # Convolutive, batch, for 2 microphones
    conv_batch_2 = {
        'name': 'Convolutive_batch_2',
        'mix_type': 'convolutive',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'batch',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': 0.05,
        'data_sets': data_sets,
        'options': {'rt60': 0.9, 'room_dim': [6.4, 3.7, 3.4], 'sigma2_awgn': 0, 'volume_gain': 5000},
        'algs': [
            {
                'name': 'AUXIVA_5', 'func': auxiva, 'state': {}, 'options': {'stft_size': 1024, 'iter': 5}
            },
            {
                'name': 'ILRMA_5', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024, 'iter': 5}
            },
            {
                'name': 'AIRES_5', 'func': AIRES_batch, 'state': {}, 'options': {'iter': 5}
            },
            {
                'name': 'AUXIVA_20', 'func': auxiva, 'state': {}, 'options': {'stft_size': 1024, 'iter': 20}
            },
            {
                'name': 'ILRMA_20', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024, 'iter': 20}
            },
            {
                'name': 'AIRES_20', 'func': AIRES_batch, 'state': {}, 'options': {'iter': 20}
            },
            {
                 'name': 'AUXIVA_50', 'func': auxiva, 'state': {}, 'options': {'stft_size': 1024, 'iter': 50}
            },
            {
                'name': 'ILRMA_50', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024, 'iter': 50}
            },
            {
                'name': 'AIRES_50', 'func': AIRES_batch, 'state': {}, 'options': {'iter': 50}
            }
        ]
    }

    # Convolutive, online, for 2 microphones
    conv_online_2 = {
        'name': 'Convolutive_online_2',
        'mix_type': 'convolutive',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'online',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': 0.05,
        'data_sets': data_sets,
        'chunk_size': 2048,
        'options': {'rt60': 0.9, 'room_dim': [6.4, 3.7, 3.4], 'sigma2_awgn': 0, 'volume_gain': 5000},
        'algs': [
            {
                'name': 'AUXIVA_1024_5', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048}
            },
            {
                'name': 'AUXIVA_2048', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048}
            },
            {
                'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}
            },
            {
                'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}
            },
            {
                'name': 'AIRES (rtap-grad)', 'func': AIRES_rtap, 'state': {}, 'options': {'type': 'grad'}
            },
            {
                'name': 'AIRES (online)', 'func': AIRES_online, 'state': {}, 'options': {}
            }
        ]
    }

    expt_batch_2 = {
        'name': 'Experimental_Batch_2',
        'mix_type': 'experimental',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'batch',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': 0.05,
        'data_sets': data_sets,
        'chunk_size': 2048,
        'options': {'rt60': 0.9, 'room_dim': [6.4, 3.7, 3.4], 'sigma2_awgn': 0, 'volume_gain': 5000},
        'algs': [
            {
                'name': 'AUXIVA_1024', 'func': auxiva, 'state': {}, 'options': {'stft_size': 1024}
            },
            {
                'name': 'AUXIVA_2048', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048}
            },
            {
                'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}
            },
            {
                'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}
            },
            {
                'name': 'AIRES (rtap-opt)', 'func': AIRES_rtap, 'state': {}, 'options': {'type': 'opt'}
            },
            #TODO:fix
            {
                'name': 'AIRES (batch)', 'func': AIRES_old, 'state': {}, 'options': {}
            }
        ]
    }

    expt_online_2 = {
        'name': 'Experimental_Online_2',
        'mix_type': 'experimental',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'online',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': 0.05,
        'data_sets': data_sets,
        'chunk_size': 2048,
        'options': {'rt60': 0.9, 'room_dim': [6.4, 3.7, 3.4], 'sigma2_awgn': 0, 'volume_gain': 5000},
        'algs': [
            {
                'name': 'AUXIVA_1024', 'func': auxiva, 'state': {}, 'options': {'stft_size': 1024}
            },
            {
                'name': 'AUXIVA_2048', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048}
            },
            {
                'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}
            },
            {
                'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}
            },
            {
                'name': 'AIRES (rtap-grad)', 'func': AIRES_rtap, 'state': {}, 'options': {'type': 'grad'}
            },
            {
                'name': 'AIRES (online)', 'func': AIRES_online, 'state': {}, 'options': {}
            }
        ]
    }

    sims = [
        conv_batch_2,
        #conv_batch_2,
    ]
    return SDRR, SARR, SIRR, sims, data_sets
