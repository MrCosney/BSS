# Joint entropy estimation, mutual information estimation,
# and optimization for Independent Component Analysis using minimized mutual information,
# to obtain directly an unmixing matrix out of the optimization!
# Including a delay for the mising process, with a pure delay.
# assumed Mixing matrix:[[1, a0*delay0],[a1*delay1, 1]]
# inverse: 1/(1-a0*delay0*a1*delay1) * [[1, -a0*delay0],[-a1*delay1, 1]]
# Gerald Schuller, October 2017

import numpy as np
import scipy.optimize as opt
import scipy.signal


def frac_delay_filt(tau):
    import scipy
    import scipy.signal as sp

    L = max(1, int(tau))
    n = np.arange(0, L)

    a_0 = np.array([1.0])
    a = np.array(np.cumprod(np.divide(np.multiply((L - n), (L - n - tau)), (np.multiply((n + 1), (n + 1 + tau))))))
    a = np.append(a_0, a)  # Denumerator of the transfer function

    b = np.flipud(a)  # Numerator of the transfer function

    Nh = L * 5  # Length of the transfer function
    impulse_train = np.zeros(Nh + 1)
    impulse_train[0] = 1
    h = scipy.signal.lfilter(b, a, impulse_train)  # Transfer function

    return a, b, h


def entropy(x, bins):
    Nanindex = np.argwhere(np.isnan(x))
    if any(Nanindex):
        print("Not a number error, index=", Nanindex)
    hist, binedges = np.histogram(x, bins)
    pdf = 1.0 * hist / np.sum(hist)
    entr = - np.sum(pdf * np.log(pdf + 1e-6))
    return entr


def entropy2d(x, bins):
    hist2d, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], bins)
    pdf2d = 1.0 * hist2d / np.sum(hist2d)
    jointentropy = - np.sum(pdf2d * np.log(pdf2d + 1e-6))
    return jointentropy


def mutualinformation(x, bins):
    HX = entropy(x[:, 0], bins)  # entropy of x
    HY = entropy(x[:, 1], bins)  # entropy of Y
    HXY = entropy2d(x, bins)  # joint entropy of X and Y
    return HX + HY - HXY


def delayfilt(delay):
    delayfilt = np.sinc(np.arange(100) - delay)
    sinwin = np.sin(np.pi / 100 * (np.arange(100) + 0.5 + 50 - delay))
    sinwin = np.clip(sinwin, 0, None)
    delayfilt *= sinwin
    return delayfilt


def unmixing(coeffs, X):
    X_del = np.zeros(X.shape)
    a = coeffs[0:2]
    delay = coeffs[2:4]

    a0, b0, h0 = frac_delay_filt(delay[0])
    a1, b1, h1 = frac_delay_filt(delay[1])

    X_del[:, 0] = X[:, 0] - a[0] * scipy.signal.lfilter(b0, a0, X[:, 1])
    X_del[:, 1] = X[:, 1] - a[1] * scipy.signal.lfilter(b1, a1, X[:, 0])

    return X_del


def mutualinfoangles(angles, X):
    alpha = angles[0]
    beta = angles[1]
    A = np.array([[np.cos(alpha), np.sin(alpha)], [np.cos(beta), np.sin(beta)]]).T
    X_prime = np.dot(X, A)
    mutualinfo = mutualinformation(X_prime, 100)
    return mutualinfo


def mutualinfocoeffs(coeffs, X):
    X_del = unmixing(coeffs, X)
    mutualinfo = mutualinformation(X_del, 100)

    return mutualinfo


def minfftklcoeffs(coeffs, X):
    X_prime = unmixing(coeffs, X)
    X_fft = np.abs(np.fft.rfft(X_prime, axis=0))

    X_fft[:, 0] = X_fft[:, 0] / np.sum(X_fft[:, 0])
    X_fft[:, 1] = X_fft[:, 1] / np.sum(X_fft[:, 1])

    fftkl = np.sum(X_fft[:, 0] * np.log((X_fft[:, 0] + 1e-6) / (X_fft[:, 1] + 1e-6)))

    return -fftkl


def minabsklcoeffs(coeffs, X):
    X_prime = unmixing(coeffs, X)
    X_abs = np.abs(X_prime)
    X_abs[:, 0] = X_abs[:, 0] / np.sum(X_abs[:, 0])
    X_abs[:, 1] = X_abs[:, 1] / np.sum(X_abs[:, 1])

    abskl = np.sum(X_abs[:, 0] * np.log((X_abs[:, 0] + 1e-6) / (X_abs[:, 1] + 1e-6)))
    return -abskl


def itakurasaitocoeffs(coeffs, X):
    X_prime = unmixing(coeffs, X)
    X_abs = np.abs(X_prime)

    X_abs[:, 0] = X_abs[:, 0] / np.sum(X_abs[:, 0])
    X_abs[:, 1] = X_abs[:, 1] / np.sum(X_abs[:, 1])

    itakurasaito = np.sum(
        X_abs[:, 0] / (X_abs[:, 1] + 1e-6) - np.log((X_abs[:, 0] + 1e-6) / (X_abs[:, 1] + 1e-6)) - 1) / 100000
    return -itakurasaito


def meanpower(coeffs, X):
    X_prime = unmixing(coeffs, X)
    power = np.sum(X_prime ** 2)
    return power


def playsound(audio, samplingRate, channels):
    import pyaudio
    p = pyaudio.PyAudio()
    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)

    audio = np.clip(audio, -2 ** 15, 2 ** 15 - 1)
    sound = (audio.astype(np.int16).tostring())
    # sound = (audio.astype(np.int16).tostring())
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return


def shullers_method(mixed: np.array, state: dict, options):
    # Prepare mixed
    mixed = mixed.T * 0.5 / 2 ** 15

    X = np.zeros(mixed.shape)

    coefficients = state['coeffs'] if 'coeffs' in state else np.array([0.9, 0.9, 5, 5])

    filter_delay_0 = delayfilt(coefficients[2])
    filter_delay_1 = delayfilt(coefficients[3])
    X[:, 0] = 1.0 * mixed[:, 0] + coefficients[0] * scipy.signal.lfilter(filter_delay_0, [1], mixed[:, 1])
    X[:, 1] = 1.0 * mixed[:, 1] + coefficients[1] * scipy.signal.lfilter(filter_delay_1, [1], mixed[:, 0])

    N = 8  # down sample by N, can be changed
    lp = scipy.signal.remez(63, bands=[0, 0.32 / N, 0.45 / N, 1], desired=[1, 0], weight=[1, 100], Hz=2)

    Xlp = scipy.signal.lfilter(lp, [1], X, axis=0)
    Xlp = Xlp[::N, :]
    Xorig = X.copy()
    X = Xlp

    bounds = [(0.0 + 1e-4, 1.0 - 1e-4), (0.0 + 1e-4, 1.0 - 1e-4), (0, 15.0), (0, 15.0)]

    # Compute separating coefficients
    coeffs_minimized = opt.minimize(minabsklcoeffs,
                                    x0=np.array([0.0, 0.0, 0.0, 0.0]),
                                    args=(X,),
                                    bounds=bounds,
                                    method='SLSQP',
                                    options={'disp': False, 'maxiter': 200})

    coefficients = coeffs_minimized.x
    coefficients[2:4] *= N

    # Store coefficients to state
    state['coeffs'] = coefficients

    X_del = unmixing(coefficients, Xorig)
    X_del = X_del * 1.0 / np.max(abs(X_del))

    return X_del.T, state
