from Player import load_wav
import numpy as np
from scipy import signal

def exctract():
    freq = 44100
    n_samples = 10000
    audio_1 = "Audio/Original/Kunkka.wav"
    audio_2 = "Audio/Original/Ench.wav"
    audio_3 = "Audio/Original/Timber.wav"

    #audio_1 = "Audio/Original/piano.wav"
    #audio_2 = "Audio/Original/drum.wav"
    #audio_3 = "Audio/Original/guitar.wav"
    '''Extract data from wav. files'''
    rate1, data1 = load_wav(audio_1, freq)
    rate2, data2 = load_wav(audio_2, freq)
    rate3, data3 = load_wav(audio_3, freq)
    source_data = np.array([data1, data2, data3])
    ''''Create simple signals.'''
    np.random.seed(0)
    time = np.linspace(0, 8, n_samples)
    sig1 = np.sin(2 * time)  # Simple sin signal
    sig2 = np.sign(np.sin(3 * time))  # Simple square signal
    sig3 = signal.sawtooth(2 * np.pi * time)  # Simple saw signal
    source_sig = np.array([sig1, sig2, sig3])
    return rate1, source_data, source_sig