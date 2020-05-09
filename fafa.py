import numpy as np
import matplotlib.pyplot as plt
from shogun import RealFeatures
from shogun import Jade
from scipy.io import wavfile
from sklearn.decomposition import FastICA, PCA
from scipy.io.wavfile import write
from Player import *

# #############################################################################
rate, data = wavfile.read('Oleg.wav')
data = data.T
unmix_folder = "Audio/UnmixedOleg/"
play(data[0])
write("".join((unmix_folder, 'OlegOriginal.wav')), rate, np.float32(data[0] * 4000))
write("".join((unmix_folder, 'OlegOriginal2.wav')), rate, np.float32(data[1] * 4000))

data = np.float64(data)
ica = FastICA(n_components=2)
S_ = ica.fit_transform(data)
A_ = ica.mixing_
S_ = S_.T
unmix_folder = "Audio/UnmixedOleg/"
write("".join((unmix_folder, 'Oleg.wav')), rate, np.float32(S_[0]) * 2000)
write("".join((unmix_folder, 'Shuler.wav')), rate, np.float32(S_[1]) * 2000)

mixed_signals = RealFeatures(data.astype(np.float64))
jade = Jade()
signals = jade.apply(mixed_signals)
JUnmixAudio = signals.get_feature_matrix()
Mix_matrix = jade.get_mixing_matrix()
Mix_matrix / Mix_matrix.sum(axis=0)

write("".join((unmix_folder, 'OlegJade.wav')), rate, np.float32(JUnmixAudio[0]))
write("".join((unmix_folder, 'ShulerJade.wav')), rate, np.float32(JUnmixAudio[1]))
