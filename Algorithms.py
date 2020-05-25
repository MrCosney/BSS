from shogun import RealFeatures
from shogun import Jade
import numpy as np
from sklearn.decomposition import FastICA, PCA
import pyroomacoustics as pra
from Player import play
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_images

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


def jade_unmix(mix_audio: np.array, state: dict, options: dict):
    mixed_signals = RealFeatures(mix_audio.astype(np.float64))
    jade = Jade()
    signals = jade.apply(mixed_signals)
    JUnmixAudio = signals.get_feature_matrix()
    Mix_matrix = jade.get_mixing_matrix()
    Mix_matrix / Mix_matrix.sum(axis=0)
    state = Mix_matrix
    return JUnmixAudio, state

def Pca(mix_audio, state: dict, options: dict):
    mix_audio = mix_audio.T
    pca = PCA(n_components=mix_audio.shape[1])
    unmix = pca.fit_transform(mix_audio)
    return unmix.T, state

def Fast(mix_audio: np.array, state: dict, options: dict):
    mix_audio = mix_audio.T
    ica = FastICA(n_components=mix_audio.shape[1])
    S_ = ica.fit_transform(mix_audio)
    Mix_matrix = ica.mixing_
    # Mix_matrix / Mix_matrix.sum(axis=0)
    state = Mix_matrix
    return S_.T, state

def convergence_callback(Y):
    ref = np.moveaxis(separate_recordings, 1, 2)
    y = np.array([pra.istft(Y[:, :, ch], L, L,
            transform=np.fft.irfft, zp_front=L//2, zp_back=L//2) for ch in range(Y.shape[2])])
    sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1]-L//2, 0], y[:, L//2:ref.shape[1]+L//2])
    SDR.append(sdr)
    SIR.append(sir)

def auxvia(mix_audio: np.array, state: dict, options: dict):
    global L, SDR, SIR, separate_recordings       #L -fft_size
    SDR, SIR = [], []
    L = options['stft_size']
    print(L)
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2) for ch in mix_audio])
    X = np.moveaxis(X, 0, 2)

    #SDR, SIR = [], []
    #TODO: Check for back projection of 1st mic "proj_back"
    Y, W = pra.bss.auxiva(X, n_iter=20, proj_back=False, return_filters=True)  #callback=convergence_callback  check for callback
    unmix = np.array([pra.istft(Y[:, :, ch], L, L, transform=np.fft.irfft, zp_front=L // 2, zp_back=L // 2) for ch in
                  range(Y.shape[2])])

    #TODO: STFT For filter state (W) and add to state
    #sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1] - L // 2, 0], y[:, L // 2:ref.shape[1] + L // 2])
    return unmix, state


def ILRMA(mix_audio: np.array, state: dict, options: dict):
    global L, SDR, SIR, separate_recordings  # L -fft_size
    SDR, SIR = [], []
    L = options['stft_size']
    print(L)
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2) for ch in mix_audio])
    X = np.moveaxis(X, 0, 2)

    # SDR, SIR = [], []
    Y, W = pra.bss.ilrma(X, n_iter=20, proj_back=False, return_filters=True)  # callback=convergence_callback  check for callback
    unmix = np.array([pra.istft(Y[:, :, ch], L, L, transform=np.fft.irfft, zp_front=L // 2, zp_back=L // 2) for ch in
                      range(Y.shape[2])])
    # TODO: STFT For filter state (W) and add to state
    # sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1] - L // 2, 0], y[:, L // 2:ref.shape[1] + L // 2])
    return unmix, state