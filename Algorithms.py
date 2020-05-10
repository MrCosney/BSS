from shogun import RealFeatures
from shogun import Jade
import numpy as np
from sklearn.decomposition import FastICA


def mix(data):
    length = 0
    for i in range(data.shape[0]):
        if len(data[i]) > length:
            length = len(data[i])

    for i in range(data.shape[0]):
        data[i].resize((length, 1), refcheck=False)
    '''Adding interference to signals'''
    # if len(s1) == 10000:
    #	S += 0.1 * np.random.normal(size=S.shape)
    '''Create mix matrix'''
    if data.shape[0] == 2:
        S = (np.c_[data[0], data[1]]).T  # TODO: fix this clipping part for multiply sources=> more flex.
        A = np.array([[1, 0.5],
                      [0.5, 1]])
    else:
        S = (np.c_[data[0], data[1], data[2]]).T  # TODO: check prev.
        A = np.array([[1, 0.5, 0.5],
                      [0.5, 1, 0.5],
                      [0.5, 0.5, 1]])
    mixed = np.dot(A, S)
    return mixed


def jade_unmix(mix_audio):
    mixed_signals = RealFeatures(mix_audio.astype(np.float64))
    jade = Jade()
    signals = jade.apply(mixed_signals)
    JUnmixAudio = signals.get_feature_matrix()
    Mix_matrix = jade.get_mixing_matrix()
    Mix_matrix / Mix_matrix.sum(axis=0)
    print("Estimated Mixing Matrix is: ")
    print(Mix_matrix)
    return JUnmixAudio


def Fast(mix_audio):
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(mix_audio)
    Mix_matrix = ica.mixing_
    # Mix_matrix / Mix_matrix.sum(axis=0)
    print("Estimated Mixing Matrix is: ")
    print(Mix_matrix)
    return S_.T
