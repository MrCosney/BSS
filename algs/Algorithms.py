#from shogun import RealFeatures
#from shogun import Jade
import numpy as np
from sklearn.decomposition import FastICA, PCA
import pyroomacoustics as pra
from Player import play
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_images, bss_eval_sources


# def jade_unmix(mix_audio: np.array, state: dict, options: dict):
#     mixed_signals = RealFeatures(mix_audio.astype(np.float64))
#     jade = Jade()
#     signals = jade.apply(mixed_signals)
#     JUnmixAudio = signals.get_feature_matrix()
#     Mix_matrix = jade.get_mixing_matrix()
#     Mix_matrix / Mix_matrix.sum(axis=0)
#     state = Mix_matrix
#     return JUnmixAudio, state


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


#def convergence_callback(Y):
#    ref = np.moveaxis(separate_recordings, 1, 2)
#    y = np.array([pra.istft(Y[:, :, ch], L, L,
#            transform=np.fft.irfft, zp_front=L//2, zp_back=L//2) for ch in range(Y.shape[2])])
#    sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1]-L//2, 0], y[:, L//2:ref.shape[1]+L//2])
#    SDR.append(sdr)
#    SIR.append(sir)


def auxvia(mix_audio: np.array, state: dict, options: dict):
    '''I rewrite it to use windows, now separation more clear but calc time is increased'''
    L = options['stft_size']
    #TODO: find a correct window and hop (L//x)
    win_a = pra.hann(L)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, L//4)
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = pra.transform.stft.analysis(mix_audio.T, L, L//4, win=win_a)
    Y = pra.bss.auxiva(X, n_iter=5)
    unmix = pra.transform.stft.synthesis(Y, L, L//4, win=win_s).T
    return unmix, state


def ILRMA(mix_audio: np.array, state: dict, options: dict):
    L = options['stft_size']
    win_a = pra.hann(L)
    win_s = pra.transform.stft.compute_synthesis_window(win_a, L // 4)
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = pra.transform.stft.analysis(mix_audio.T, L, L // 4, win=win_a)
    Y = pra.bss.ilrma(X, n_iter=5, proj_back=True)
    unmix = pra.transform.stft.synthesis(Y, L, L // 4, win=win_s).T
    return unmix, state
