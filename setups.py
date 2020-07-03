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
            #'data': ["Audio/Original/Man.wav", "Audio/Original/Woman.wav", "Audio/Original/Announcer.wav"],
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

    sims = [
        # Convolutive for 2 microphones absorp: 0.35, orders: 0, awgn: 0
        {
            'name': 'Convolutive_3',
                        #'mix_type': 'linear', 'convolutive', 'convolutive_real'
            'mix_type': 'convolutive',
            'microphones': 3,
            'data_sets': data_sets,
            'chunk_size': 2048,
            # 'options': {'absorp': 0.35, 'orders': 0, 'sigma2_awgn': 0},
            'options': {'rt60': 0.5, 'room_dim': [7, 5, 3.2], 'sigma2_awgn': 0},
            'algs': [
                {
                    'name': 'AUXIVA_512', 'func': auxiva, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_1024', 'func': auxiva, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'AUXIVA_2048', 'func': auxiva, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_512', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 512}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_1024', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 1024}, 'metrics': {}
                },
                {
                    'name': 'ILRMA_2048', 'func': ILRMA, 'state': {}, 'options': {'stft_size': 2048}, 'metrics': {}
                },
                {
                    'name': 'AIRES', 'func': shullers_method, 'state': {}, 'metrics': {}
                }
            ]
        }
    ]
    return sims, data_sets


def speakers_device_idx():
    """This functions get device index for output source data to speakers"""
    import pyaudio
    import sys

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
