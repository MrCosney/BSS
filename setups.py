from Algorithms import *
from Olegs import shullers_method
from scipy import signal


def setups():
    np.random.seed(0)
    n_samples = 10000
    time = np.linspace(0, 8, n_samples)
    data_sets = [
        {
            'type': 'Voice',
            'freq': 44100,
            'data': [
                "Audio/Original/Kunkka.wav",
                "Audio/Original/Ench.wav",
                "Audio/Original/Timber.wav"
            ]
        },
        {
            'type': 'Music',
            'freq': 44100,
            'data': [
                "Audio/Original/piano.wav",
                "Audio/Original/drum.wav",
                "Audio/Original/guitar.wav"
            ]
        },
        {
            'type': 'Gen Signals',
            'freq': 44100,
            'data': [
                np.sin(2 * time),
                np.sign(np.sin(3 * time)),
                signal.sawtooth(2 * np.pi * time)
            ]
        }
    ]

    sims = [
        {
            'name': 'Linear_2',
            'mixture_type': 'linear',
            'microphones': 2,
            'data_sets': data_sets,
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_2',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_2',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_2',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AIRES',
                    'func': shullers_method,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        },
                        #Linear for 3 microphones
        {
            'name': 'Linear_3',
            'mixture_type': 'linear',
            'microphones': 3,
            'data_sets': data_sets,
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_3',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_3',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_3',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        },
            # Convolutive for 2 microphones absorp: 0.35, orders: 0, awgn: 0
        {
            'name': 'Convolutive_2_0',
            'mixture_type': 'room',
            'microphones': 2,
            'data_sets': data_sets,
            'options': {'absorp': 0.35, 'orders': 0, 'awgn': 0},
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_2',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_2',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_2',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AIRES',
                    'func': shullers_method,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        },
            # Convolutive for 2 microphones absorp: 0.35, orders: 7, awgn: 0
        {
            'name': 'Convolutive_2_7',
            'mixture_type': 'room',
            'microphones': 2,
            'data_sets': data_sets,
            'options': {'absorp': 0.35, 'orders': 7, 'awgn': 0},
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_2',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_2',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_2',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AIRES',
                    'func': shullers_method,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        },
            # Convolutive for 2 microphones absorp: 0.35, orders: 7, awgn: 1e-8
        {
            'name': 'Convolutive_2_7_n',
            'mixture_type': 'room',
            'microphones': 2,
            'data_sets': data_sets,
            'options': {'absorp': 0.35, 'orders': 7, 'awgn': 1e-8},
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_2',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_2',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_2',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AIRES',
                    'func': shullers_method,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        },
            # Convolutive for 3 microphones absorp: 0.35, orders: 0, awgn: 0
        {
            'name': 'Convolutive_3_0',
            'mixture_type': 'room',
            'microphones': 3,
            'data_sets': data_sets,
            'options': {'absorp': 0.35, 'orders': 0, 'awgn': 0},
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_2',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_2',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_2',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AIRES',
                    'func': shullers_method,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        },
            # Convolutive for 3 microphones absorp: 0.35, orders: 7, awgn: 0
        {
            'name': 'Convolutive_3_7',
            'mixture_type': 'room',
            'microphones': 3,
            'data_sets': data_sets,
            'options': {'absorp': 0.35, 'orders': 7, 'awgn': 0},
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_2',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_2',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_2',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AIRES',
                    'func': shullers_method,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        },
            # Convolutive for 3 microphones absorp: 0.35, orders: 7, awgn: 0
        {
            'name': 'Convolutive_3_7_n',
            'mixture_type': 'room',
            'microphones': 3,
            'data_sets': data_sets,
            'options': {'absorp': 0.35, 'orders': 7, 'awgn': 1e-8},
            'mix_data': {},
            'algs': [
                {
                    'name': 'JADE_2',
                    'func': jade_unmix,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'PCA_2',
                    'func': Pca,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'ICAA_2',
                    'func': Fast,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AIRES',
                    'func': shullers_method,
                    'state': {},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_512',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'AUXIVA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_512',
                    'func': ILRMA,
                    'state': {},
                    'options': {'stft_size': 512},
                    'Metrics': {}
                },
                {
                    'name': 'ILRMA_2048',
                    'func': auxvia,
                    'state': {},
                    'options': {'stft_size': 2048},
                    'Metrics': {}
                }
            ]
        }
    ]
    return sims, data_sets
