#from shogun import RealFeatures
#from shogun import Jade
import numpy as np
from sklearn.decomposition import FastICA, PCA
import pyroomacoustics as pra


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

    #TODO: Оба алгоритма ниже практически одинаковы,но пока не переносил в единую функцию, с выбором алогритма по имени
    # возможно какие то параметры будуем менять у каждого отдельно.

def auxiva(mix_audio: np.array, state: dict, options: dict):
    if 'stft' not in state:
        L = options['stft_size']
        hop = L // 2
        window = pra.hann(L, flag='asymmetric', length='full')
        stft = pra.transform.STFT(L, hop=hop, analysis_window=window, channels=mix_audio.shape[0])
    else:
        stft = state['stft']
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = stft.analysis(mix_audio.T)
    if "Filter_state" in state:
        try:
            Y, filter_state = pra.bss.auxiva(X, n_iter=5, W0=state['Filter_state'], return_filters=True)
        except:
            return mix_audio, state
       # stft = pra.transform.STFT(L, hop=hop, analysis_window=window, channels=mix_audio.shape[0])
    else:
        try:
            Y, filter_state = pra.bss.auxiva(X, n_iter=5, return_filters=True)
        except:
            return mix_audio, state
    unmix = stft.synthesis(Y)

    #save filters state and overlap data
    state['Filter_state'] = filter_state
    state['stft'] = stft
    return unmix.T, state


def ILRMA(mix_audio: np.array, state: dict, options: dict):
    if 'stft' not in state:
        L = options['stft_size']
        hop = L // 2
        window = pra.hann(L, flag='asymmetric', length='full')
        stft = pra.transform.STFT(L, hop=hop, analysis_window=window, channels=mix_audio.shape[0])
    else:
        stft = state['stft']
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = stft.analysis(mix_audio.T)
    if "Filter_state" in state:
        try:
            Y, filter_state = pra.bss.ilrma(X, n_iter=5, W0=state['Filter_state'], return_filters=True, proj_back=True)
        except:
            return mix_audio, state
    # stft = pra.transform.STFT(L, hop=hop, analysis_window=window, channels=mix_audio.shape[0])
    else:
        try:
            Y, filter_state = pra.bss.ilrma(X, n_iter=5, return_filters=True, proj_back=True)
        except:
            return mix_audio, state
    unmix = stft.synthesis(Y)

    # save filters state and overlap data
    state['Filter_state'] = filter_state
    state['stft'] = stft
    return unmix.T, state

