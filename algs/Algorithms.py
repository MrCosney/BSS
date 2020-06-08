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
    win_s = pra.transform.compute_synthesis_window(win_a, L//4)
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = pra.transform.analysis(mix_audio.T, L, L//4, win=win_a)
    Y = pra.bss.auxiva(X, n_iter=5)
    unmix = pra.transform.synthesis(Y, L, L//4, win=win_s).T

    #X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2) for ch in mix_audio])
    #X = np.moveaxis(X, 2, 0)
    #Y = np.moveaxis(Y, 2, 0)
    #X = np.moveaxis(X, 1, 2)
    #Y = np.moveaxis(Y, 1, 2)
    #TODO: next lines represent 3 ways for SDR and SIR calculation. It uses Tensors as the input. I tried to use ref and unmix sig in freq domain but shape isn't correct
    #sdr, isr, sir, sar, perm = bss_eval_sources(X, Y)
    #sdr, isr, sir, sar, perm = bss_eval_images(X[:, :Y.shape[1] - L // 2, 0], Y[:, L // 2:X.shape[1] + L // 2])
    # m = np.minimum(Y.shape[1], X.shape[1])
    # sdr, sir, sar, perm = bss_eval_sources(X[:, :m], Y[:, :m])

    # Y, W = pra.bss.auxiva(X, n_iter=20, proj_back=False, return_filters=True)# callback=convergence_callback  check for callback
    #unmix = np.array([pra.istft(Y[:, :, ch], L, L, transform=np.fft.irfft, zp_front=L // 2, zp_back=L // 2) for ch in
     #             range(Y.shape[2])])

    #m = np.minimum(Y.shape[1], X.shape[1])
    #sdr, sir, sar, perm = bss_eval_sources(X[:, :m], Y[:, :m])
    #sdr, isr, sir, sar, perm = bss_eval_images(X[:, :Y.shape[1] - L // 2, 0], Y[:, L // 2:X.shape[1] + L // 2])
    return unmix, state


def ILRMA(mix_audio: np.array, state: dict, options: dict):
    L = options['stft_size']
    win_a = pra.hann(L)
    win_s = pra.transform.compute_synthesis_window(win_a, L // 4)
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = pra.transform.analysis(mix_audio.T, L, L // 4, win=win_a)
    Y = pra.bss.ilrma(X, n_iter=5, proj_back=True)
    unmix = pra.transform.synthesis(Y, L, L // 4, win=win_s).T
    #unmix = np.array([pra.istft(Y[:, :, ch], L, L, transform=np.fft.irfft, zp_front=L // 2, zp_back=L // 2) for ch in range(Y.shape[2])])
    #play(unmix[1] * 5000)

    # TODO: STFT For filter state (W) and add to state
    # sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1] - L // 2, 0], y[:, L // 2:ref.shape[1] + L // 2])
    return unmix, state
