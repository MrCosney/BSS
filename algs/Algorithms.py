# from shogun import RealFeatures
# from shogun import Jade
import numpy as np
from sklearn.decomposition import FastICA as skFastICA
from sklearn.decomposition import PCA as skPCA
import pyroomacoustics as pra
from algs.aires.AIRES_old import shullers_method
import scipy.optimize as opt
from algs.aires.AIRES_rtap import lp_filter_and_downsample, find_coeffs_optimization, cost_n_abs_kl, unmixing
from algs.aires.aires_bss_lib_classes import offline_aires_separation, aires_online_class
from MatlabUtils import *


def AIRES_new_online(mixed, state: dict, options: dict):
    # Get aires object
    if 'aires' in state:
        aires = state['aires']
    else:
        aires = aires_online_class()
        aires.maxdelay = options['max_delay'] if 'max_delay' in options else 20
        # Number of previous blocks in memory
        aires.n_blocks_signal_memory = options['blocks_memory'] if 'blocks_memory' in options else 1
        aires.Blocksize = mixed.shape[1]  # Size of the processed block
        aires.n_iter_pro_block = options['iter_p_block'] if 'iter_p_block' in options else 2
        aires.reset_parameters()

    state['aires'] = aires

    # Unmixing
    unmixed, p_time = aires.aires_online_separation(mixed[0:2, :].T, rt60=0.2)

    return unmixed.T, state


def AIRES_new_offline(mixed, state: dict, options: dict):
    unmixed, p_time = offline_aires_separation(mixed.T, options)
    return unmixed.T, state


def AIRES_old_offline(mixed, state: dict, options: dict):
    if mixed.shape[0] < 2:
        raise ValueError('AIRES_rtap -> at least 2 channels needed for separation')

    # ESTIMATION ====>

    # Select first two microphones
    X = mixed[0:2, :]

    # Input data matrix X
    X = (X * 1.0 / np.max(abs(X))).transpose()  # now it's CHUNK x CHANNELS matrix

    # Low-pass filtering first
    N = 8
    X_lp, state['lp_filter_state'] = lp_filter_and_downsample(X,
                                                              N,
                                                              state['lp_filter_state'] if 'lp_filter_state' in state
                                                              else None)

    # Get current/initial coeffs
    coeffs = state['coeffs'] if 'coeffs' in state else np.array([0.0, 0.0, 0.0, 0.0])

    # Get estimation low-pass filter states
    filter_state_0_lp = state['filter_state_0_lp'] if 'filter_state_0_lp' in state else None
    filter_state_1_lp = state['filter_state_1_lp'] if 'filter_state_1_lp' in state else None

    # Estimate coeffs
    if options['type'] == 'opt':
        coeffs = find_coeffs_optimization(X_lp, coeffs, False)
    elif options['type'] == 'grad':
        eps = np.sqrt(np.finfo(float).eps)
        if filter_state_0_lp is None or filter_state_1_lp is None:
            gradient = opt.approx_fprime(coeffs, cost_n_abs_kl, eps, X_lp)
        else:
            gradient = opt.approx_fprime(coeffs, cost_n_abs_kl, eps, X_lp, filter_state_0_lp, filter_state_1_lp)
        # Update coeffs
        coeffs = coeffs - 0.1 * gradient
    else:
        raise ValueError("AIRES_rtap -> you have to specify valid type (\'opt\' or \'grad\')")

    # Update low-pass filter states
    if filter_state_0_lp is None or filter_state_1_lp is None:
        X_unm, state['filter_state_0_lp'], state['filter_state_1_lp'] = unmixing(coeffs, X_lp)
    else:
        X_unm, state['filter_state_0_lp'], state['filter_state_1_lp'] = unmixing(coeffs, X_lp,
                                                                                 filter_state_0=filter_state_0_lp,
                                                                                 filter_state_1=filter_state_1_lp)

    # Update coeffs
    state['coeffs'] = coeffs

    # UNMIXING ====>
    coeffs_undownsampled = coeffs.copy()
    coeffs_undownsampled[2:4] *= N

    # Get estimation low-pass filter states
    filter_state_0 = state['filter_state_0'] if 'filter_state_0' in state else None
    filter_state_1 = state['filter_state_1'] if 'filter_state_1' in state else None

    # Perform unmixing with current coefficients
    if filter_state_0 is None or filter_state_1 is None:
        X_unmixed, state['filter_state_0'], state['filter_state_1'] = unmixing(coeffs, X)
    else:
        X_unmixed, state['filter_state_0'], state['filter_state_1'] = unmixing(coeffs, X,
                                                                               filter_state_0=filter_state_0,
                                                                               filter_state_1=filter_state_1)

    return X_unmixed.T, state


def AIRES_old(mixed, state: dict, options: dict):
    if mixed.shape[0] < 2:
        raise ValueError('AIRES_old -> at least 2 channels needed for separation')

    return shullers_method(mixed, state, options)


# def jade_unmix(mixed: np.array, state: dict, options: dict):
#     mixed_signals = RealFeatures(mixed.astype(np.float64))
#     jade = Jade()
#     signals = jade.apply(mixed_signals)
#     JUnmixAudio = signals.get_feature_matrix()
#     Mix_matrix = jade.get_mixing_matrix()
#     Mix_matrix / Mix_matrix.sum(axis=0)
#     state = Mix_matrix
#     return JUnmixAudio, state


def PCA(mixed, state: dict, options: dict):
    mixed = mixed.T
    unmix = skPCA(n_components=mixed.shape[1]).fit_transform(mixed)
    return unmix.T, state


def FastICA(mixed: np.array, state: dict, options: dict):
    mixed = mixed.T
    ica = skFastICA(n_components=mixed.shape[1])
    S_ = ica.fit_transform(mixed)
    Mix_matrix = ica.mixing_
    # Mix_matrix / Mix_matrix.sum(axis=0)
    state = Mix_matrix
    return S_.T, state


# def convergence_callback(Y):
#     ref = np.moveaxis(separate_recordings, 1, 2)
#     y = np.array([pra.istft(Y[:, :, ch], L, L,
#             transform=np.fft.irfft, zp_front=L//2, zp_back=L//2) for ch in range(Y.shape[2])])
#     sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1]-L//2, 0], y[:, L//2:ref.shape[1]+L//2])
#     SDR.append(sdr)
#     SIR.append(sir)

    # TODO: Оба алгоритма ниже практически одинаковы,но пока не переносил в единую функцию, с выбором алогритма по имени
    # возможно какие то параметры будуем менять у каждого отдельно.

def AuxIVA(mixed: np.array, state: dict, options: dict):
    # Get STFT object
    if 'stft' not in state:
        stft = create_stft(mixed.shape[0], options)
        state['stft'] = stft
    else:
        stft = state['stft']
    # Compute stft
    X = stft.analysis(mixed.T)
    # Run AUXIVA
    try:
        Y, state['W0'] = pra.bss.auxiva(X,
                                        n_iter=options['iter'],
                                        n_src=mixed.shape[0],
                                        W0=state['W0'] if 'W0' in state else None,
                                        return_filters=True)
    except:
        return mixed, state

    # Compute unmixed data
    unmixed = stft.synthesis(Y).T

    return unmixed, state


def ILRMA(mixed: np.array, state: dict, options: dict):
    # Get STFT object
    if 'stft' not in state:
        stft = create_stft(mixed.shape[0], options)
        state['stft'] = stft
    else:
        stft = state['stft']

    # Compute stft
    X = stft.analysis(mixed.T)

    # Run ILRMA
    try:
        Y, state['W0'] = pra.bss.ilrma(X,
                                       n_iter=options['iter'],
                                       n_components=options['nBases'],
                                       n_src=mixed.shape[0],
                                       W0=state['W0'] if 'W0' in state else None,
                                       return_filters=True,
                                       proj_back=True)
    except:
        return mixed, state

    # Compute unmixed data
    unmixed = stft.synthesis(Y).T

    return unmixed, state


def ILRMA_MATLAB(mixed: np.array, state: dict, options: dict):
    engine = find_engine()
    if 'W0' in state:
        unmixed_m, W0 = engine.ilrma_bss(matlab.double(initializer=mixed.T.tolist(), is_complex=True),
                                         float(options['iter']),
                                         float(options['stft_size']),
                                         float(options['nBases']),
                                         matlab.double(initializer=state['W0'].tolist(), is_complex=True),
                                         nargout=2)
    else:
        unmixed_m, W0 = engine.ilrma_bss(matlab.double(initializer=mixed.T.tolist(), is_complex=True),
                                         float(options['iter']),
                                         float(options['stft_size']),
                                         float(options['nBases']),
                                         nargout=2)

    state['W0'] = np.asarray(W0)
    return np.asarray(unmixed_m).T, state


def AuxIVA_MATLAB(mixed: np.array, state: dict, options: dict):
    engine = find_engine()
    if 'W0' in state:
        unmixed_m, W0 = engine.auxiva_bss(matlab.double(initializer=mixed.T.tolist(), is_complex=True),
                                          float(options['iter']),
                                          float(options['stft_size']),
                                          matlab.double(initializer=state['W0'].tolist(), is_complex=True),
                                          nargout=2)
    else:
        unmixed_m, W0 = engine.auxiva_bss(matlab.double(initializer=mixed.T.tolist(), is_complex=True),
                                          float(options['iter']),
                                          float(options['stft_size']),
                                          nargout=2)

    state['W0'] = np.asarray(W0)

    return np.asarray(unmixed_m).T, state


def create_stft(M: int, options: dict):
    L = options['stft_size']
    hop = L // 2
    # window = pra.hann(L, flag='asymmetric', length='full')
    window = pra.hamming(L, flag='asymmetric', length='full')  # looks like hamming window is better
    return pra.transform.STFT(L, hop=hop, analysis_window=window, channels=M)
