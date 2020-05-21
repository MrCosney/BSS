from scipy.io import wavfile
import pyroomacoustics as pra
from Player import play
import numpy as np

# input_array = [
# [1, 2, 3],
# [4, 5, 6],
# [7, 8, 9]
# ]
#
# result = np.concatenate(input_array)
# print(result)

# read multichannel wav file
# audio.shape == (nsamples, nchannels)
fs, audio = wavfile.read("Audio/Original/Oleg.wav")

# STFT analysis parameters
fft_size = 4096  # `fft_size / fs` should be ~RT60
hop = fft_size // 2  # half-overlap
win_a = pra.hann(fft_size)  # analysis window
# optimal synthesis window
win_s = pra.transform.compute_synthesis_window(win_a, hop)

# STFT
# X.shape == (nframes, nfrequencies, nchannels)
X = pra.transform.analysis(audio, fft_size, hop, win=win_a)

# Separation
Y = pra.bss.auxiva(X, n_iter=20)

# iSTFT (introduces an offset of `hop` samples)
# y contains the time domain separated signals
# y.shape == (new_nsamples, nchannels)
y = pra.transform.synthesis(Y, fft_size, hop, win=win_s)
y = y.T

play(y[1] * 10000)
