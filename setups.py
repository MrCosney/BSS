from algs.Algorithms import *


def setups():
    np.random.seed(0)
    n_samples = 20000
    time = np.linspace(0, 8, n_samples)
    data_sets = [
        {
            'type': 'Voice',
            'fs': 44100,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/Kunkka.wav", "Audio/Original/Ench.wav", "Audio/Original/Timber.wav"],
            'file_names': ["Man.wav", "Woman.wav", "Announcer.wav"]
        },
        # {
        #     'type': 'Music',
        #     'fs': 44100,
        #     'data': ["Audio/Original/piano.wav", "Audio/Original/drum.wav", "Audio/Original/guitar.wav"]
        # },
        # {
        #     'type': 'Gen Signals',
        #     'fs': 44100,
        #     'data': [np.sin(2 * time), np.sign(np.sin(3 * time)), signal.sawtooth(2 * np.pi * time)]
        # }
    ]

    # Convolutive, batch, for 2 microphones
    conv_batch_2 = {
        'name': 'Convolutive_batch_2',
        'mix_type': 'convolutive',  # 'linear', 'convolutive', 'experimental'
        'run_type': 'batch',  # 'batch', 'online'
        'sources': 2,
        'microphones': 2,
        'microphones_distance': 0.1,
        'data_sets': data_sets,
        'options': {'rt60': 0.2, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0, 'volume_gain': 5000},
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
                'name': 'AIRES (old)', 'func': AIRES_old, 'state': {}, 'options': {}
            },
            {
                'name': 'AIRES (rtap-opt)', 'func': AIRES_rtap, 'state': {}, 'options': {'type': 'opt'}
            },
            {
                'name': 'AIRES (rtap-grad)', 'func': AIRES_rtap, 'state': {}, 'options': {'type': 'grad'}
            },
            {
                'name': 'AIRES (batch)', 'func': AIRES_batch, 'state': {}, 'options': {}
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
        'microphones_distance': 0.1,
        'data_sets': data_sets,
        'chunk_size': 2048,
        'options': {'rt60': 0.2, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0, 'volume_gain': 5000},
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
                'name': 'AIRES (old)', 'func': AIRES_old, 'state': {}, 'options': {}
            },
            {
                'name': 'AIRES (rtap-opt)', 'func': AIRES_rtap, 'state': {}, 'options': {'type': 'opt'}
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
        conv_online_2
    ]
    return sims, data_sets
