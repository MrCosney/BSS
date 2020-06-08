from algs.Algorithms import *
from Olegs import shullers_method
from scipy import signal


def setups():
    np.random.seed(0)
    n_samples = 20000
    time = np.linspace(0, 8, n_samples)
    data_sets = [
        {
            'type': 'Voice',
            'fs': 44100,
            'data': ["Audio/Original/Kunkka.wav", "Audio/Original/Ench.wav", "Audio/Original/Timber.wav"]
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

    sims = [
        # Convolutive for 2 microphones absorp: 0.35, orders: 0, awgn: 0
        {
            'name': 'Convolutive_2_0',
            # 'mix_type': 'linear',
            'mix_type': 'convolutive',
            'microphones': 2,
            'data_sets': data_sets,
            # 'options': {'absorp': 0.35, 'orders': 0, 'sigma2_awgn': 0},
            'options': {'rt60': 0.5, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0},
            'algs': [
                {
                    'name': 'AIRES', 'func': shullers_method, 'state': {}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_512', 'func': auxvia, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_1024', 'func': auxvia, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_2048', 'func': auxvia, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_512', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                }
            ]
        },
            # Convolutive for 2 microphones absorp: 0.35, orders: 7, awgn: 0
        # {
        #     'name': 'Convolutive_2_7',
        #     'mix_type': 'convolutive',
        #     'microphones': 2,
        #     'data_sets': data_sets,
        #     # 'options': {'absorp': 0.35, 'orders': 0, 'sigma2_awgn': 0},
        #     'options': {'rt60': 0.5, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0},
        #     'algs': [
        #         {
        #             'name': 'AIRES', 'func': shullers_method, 'state': {}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_512', 'func': auxvia, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_1024', 'func': auxvia, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_2048', 'func': auxvia, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_512', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
        #         }
        #     ]
        # },
            # Convolutive for 2 microphones absorp: 0.35, orders: 7, awgn: 1e-8
        # {
        #     'name': 'Convolutive_2_7_n',
        #     'mix_type': 'convolutive',
        #     'microphones': 2,
        #     'data_sets': data_sets,
        #     # 'options': {'absorp': 0.35, 'orders': 0, 'sigma2_awgn': 0},
        #     'options': {'rt60': 0.5, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0},
        #     'algs': [
        #         {
        #             'name': 'AIRES', 'func': shullers_method, 'state': {}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_512', 'func': auxvia, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_1024', 'func': auxvia, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_2048', 'func': auxvia, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_512', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
        #         }
        #     ]
        # },
            # Convolutive for 3 microphones absorp: 0.35, orders: 0, awgn: 0
        {
            'name': 'Convolutive_3_0',
            'mix_type': 'convolutive',
            'microphones': 3,
            'data_sets': data_sets,
            # 'options': {'absorp': 0.35, 'orders': 0, 'sigma2_awgn': 0},
            'options': {'rt60': 0.5, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0},
            'algs': [
                {
                    'name': 'AUXIVA_512', 'func': auxvia, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_1024', 'func': auxvia, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_2048', 'func': auxvia, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_512', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                }
            ]
        },
            # Convolutive for 3 microphones absorp: 0.35, orders: 7, awgn: 0
        {
            'name': 'Convolutive_3_7',
            'mix_type': 'convolutive',
            'microphones': 3,
            'data_sets': data_sets,
            # 'options': {'absorp': 0.35, 'orders': 0, 'sigma2_awgn': 0},
            'options': {'rt60': 0.5, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0},
            'algs': [
                {
                    'name': 'AUXIVA_512', 'func': auxvia, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_1024', 'func': auxvia, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_2048', 'func': auxvia, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_512', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                }
            ]
        # },
        #     # Convolutive for 3 microphones absorp: 0.35, orders: 7, awgn: 0
        # {
        #     'name': 'Convolutive_3_7_n',
        #     'mix_type': 'convolutive',
        #     'microphones': 3,
        #     'data_sets': data_sets,
        #     # 'options': {'absorp': 0.35, 'orders': 0, 'sigma2_awgn': 0},
        #     'options': {'rt60': 0.5, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0},
        #     'algs': [
        #         {
        #             'name': 'AIRES', 'func': shullers_method, 'state': {}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_512', 'func': auxvia, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_1024', 'func': auxvia, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
        #         },
        #         {
        #             'name': 'AUXIVA_2048', 'func': auxvia, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_512', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
        #         },
        #         {
        #             'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
        #         }
        #     ]
        }
    ]
    return sims, data_sets
