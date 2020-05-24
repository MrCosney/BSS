from Algorithms import *
from Olegs import shullers_method

def setups():
    sims = [
        {
            'name': 'Linear',
            'mixture_type': 'linear'
        },
        {
            'name': 'Convolutive',
            'mixture_type': 'room'
        }
    ]

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
        }
    ]

    algs = [
            {
                'name': 'JADE_2',
                'func': jade_unmix,
                'microphones': 2,
                'state': {}
            },
            {
                'name': 'JADE_3',
                'func': jade_unmix,
                'microphones': 3,
                'state': {}
            },
            {
                'name': 'PCA_2',
                'func': Pca,
                'microphones': 2,
                'state': {}
            },
            {
                'name': 'PCA_3',
                'func': Pca,
                'microphones': 3,
                'state': {}
            },
            {
                'name': 'ICAA_2',
                'func': Fast,
                'microphones': 2,
                'state': {}
            },
            {
                'name': 'ICAA_3',
                'func': Fast,
                'microphones': 3,
                'state': {}
            },
            {
                'name': 'AIRES',
                'func': shullers_method,
                'microphones': 2,
                'state': {}
            }
        ]
    return sims, data_sets, algs