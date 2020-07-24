from algs.Algorithms import *


def setups():
    SDRR = []
    SARR = []
    SIRR = []
    np.random.seed(0)
    n_samples = 20000
    time = np.linspace(0, 8, n_samples)
    data_sets = [
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/2.wav"],
            'file_names': ["1.wav", "2.wav"]
         },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/3.wav"],
            'file_names': ["1.wav", "3.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/4.wav"],
            'file_names': ["1.wav", "4.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/5.wav"],
            'file_names': ["1.wav", "5.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/6.wav"],
            'file_names': ["1.wav", "6.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/7.wav"],
            'file_names': ["1.wav", "7.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/8.wav"],
            'file_names': ["1.wav", "8.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/9.wav"],
            'file_names': ["1.wav", "9.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/1.wav", "Audio/Original/10.wav"],
            'file_names': ["1.wav", "10.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/3.wav"],
            'file_names': ["2.wav", "3.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/4.wav"],
            'file_names': ["2.wav", "4.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/5.wav"],
            'file_names': ["1.wav", "5.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/6.wav"],
            'file_names': ["2.wav", "6.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/7.wav"],
            'file_names': ["2.wav", "7.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/8.wav"],
            'file_names': ["1.wav", "10.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/9.wav"],
            'file_names': ["1.wav", "10.wav"]
        },

        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/2.wav", "Audio/Original/10.wav"],
            'file_names': ["1.wav", "10.wav"]
        },
        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/3.wav", "Audio/Original/4.wav"],
            'file_names': ["3.wav", "4.wav"]
        },

        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/3.wav", "Audio/Original/5.wav"],
            'file_names': ["3.wav", "5.wav"]
        },

        {
            'type': 'Voice',
            'fs': 16000,
            # 'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
            'data': ["Audio/Original/3.wav", "Audio/Original/6.wav"],
            'file_names': ["3.wav", "6.wav"]
        }
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
                'name': 'AUXIVA_5', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048, 'iter': 5}
            },
            {
                 'name': 'AUXIVA_10', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048, 'iter': 10}
             },
            {
                 'name': 'AUXIVA_50', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048, 'iter': 50}
            },
            {
                'name': 'ILRMA', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}
             },
            {
                 'name': 'AIRES (rtap-opt)', 'func': AIRES_rtap, 'state': {}, 'options': {'type': 'opt'}
             }
            ,
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
