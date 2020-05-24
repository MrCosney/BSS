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


def jade_unmix(mix_audio: np.array, state: dict, options):
    mixed_signals = RealFeatures(mix_audio.astype(np.float64))
    jade = Jade()
    signals = jade.apply(mixed_signals)
    JUnmixAudio = signals.get_feature_matrix()
    Mix_matrix = jade.get_mixing_matrix()
    Mix_matrix / Mix_matrix.sum(axis=0)
    state = Mix_matrix
    return JUnmixAudio, state

def Pca(mix_audio, state: dict, options):
    mix_audio = mix_audio.T
    pca = PCA(n_components=mix_audio.shape[1])
    unmix = pca.fit_transform(mix_audio)
    return unmix.T, state

def Fast(mix_audio: np.array, state: dict, options):
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

def auxvia(mix):
    global L, SDR, SIR, separate_recordings       #L -fft_size
    SDR, SIR = [], []
    fs = 44100
    L = 4096
     # TODO:For the future Room can be returned from Room.py file
    '''Delay setup '''
    #delay = [1., 0.5, 0.]
    #delay = [1., 1., 1.]
    #delay = [1., 0.7, 0.3]
    '''Sources and mics setup'''
    room = pra.ShoeBox([7, 5, 3.2], fs=fs, absorption=0.35, max_order=15, sigma2_awgn=1e-8)     #TODO:Check for fs
    #room.add_source([3., 2., 1.8], signal=mix[0], delay=delay[0])
    #room.add_source([6., 4., 1.8], signal=mix[1], delay=delay[1])
    #room.add_source([2., 4.5, 1.8], signal=mix[2], delay=delay[2])
    room.add_source([3., 2., 1.8], signal=mix[0])
    room.add_source([6., 4., 1.8], signal=mix[1])
    room.add_source([2., 4.5, 1.8], signal=mix[2])
    '''Place the mic. array into the room'''
    R = np.c_[
        [3, 2.87, 1],  # microphone 1
        [3, 2.93, 1],  # microphone 2
        [3, 2.99, 1],  # microphone 3
    ]
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    room.compute_rir()
    '''Record each source separately'''
    signals = list(mix)
    separate_recordings = []
    for source, signal in zip(room.sources, signals):
        source.signal[:] = signal

        room.simulate()
        separate_recordings.append(room.mic_array.signals)

        source.signal[:] = 0.
    separate_recordings = np.array(separate_recordings)
    # Mix down the recorded signals
    mics_signals = np.sum(separate_recordings, axis=0)
    '''STFT Processing'''
    # Observation vector in the STFT domain
    X = np.array([pra.stft(ch, L, L, transform=np.fft.rfft, zp_front=L // 2, zp_back=L // 2) for ch in mics_signals])
    X = np.moveaxis(X, 0, 2)

    ref = np.moveaxis(separate_recordings, 1, 2)
    SDR, SIR = [], []
    Y = pra.bss.auxiva(X, n_iter=30, proj_back=True, callback=convergence_callback)
    unmix = np.array([pra.istft(Y[:, :, ch], L, L, transform=np.fft.irfft, zp_front=L // 2, zp_back=L // 2) for ch in
                  range(Y.shape[2])])

    #sdr, isr, sir, sar, perm = bss_eval_images(ref[:, :y.shape[1] - L // 2, 0], y[:, L // 2:ref.shape[1] + L // 2])
    #play(unmix[2] * 300000)
    #play(ref[0] * 300000)
    return unmix, mics_signals