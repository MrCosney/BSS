import numpy as np
import scipy
import scipy.optimize as opt
import scipy.signal


def allp_delay_filter(tau):
    """
    performs Fractional-delay All-pass Filter
    :param tau: fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
    :type tau: float or int
    :return:
        a: Denominator of the transfer function
        b: Numerator of the transfer function
        h: Transfer function of the filter
    """
    a, b, L, l = allp_ab(tau)
    # print("Denominator of the transfer function a:", a)
    # print("Numerator of the transfer function b:", b)
    # Calculate Impulse response of the filter
    Nh = L * 5  # Length of the transfer function (to show)
    # Impulse signal to output part of the impulse response
    impulse = np.zeros(Nh + 1)
    impulse[0] = 1
    h = scipy.signal.lfilter(b, a, impulse)  # impulse response
    return a, b, h


def sinc_delay_filter(tau):
    """
    Computes filter coefficients for sinc fractional-delay filter
    :param tau: fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
    :type tau: float or int
    :return:
        a: Denominator of the transfer function
        b: Numerator of the transfer function
        h: Transfer function of the filter
    """
    a, b, L, l = sinc_ab(tau, 100)
    # print("Denominator of the transfer function a:", a)
    # print("Numerator of the transfer function b:", b)
    # Calculate Impulse response of the filter
    # Impulse signal to output part of the impulse response
    impulse = np.zeros(L + 1)
    impulse[0] = 1
    h = scipy.signal.lfilter(b, a, impulse)  # impulse response
    return a, b, h


def sinc_ab(tau, L):
    l = np.arange(L)
    # Denominator is 1 - FIR
    a = np.array([1])
    b = np.sinc(l - tau)
    # Window
    sin_win = np.sin(np.pi / 100 * (l + 0.5 + 50 - tau))
    sin_win = np.clip(sin_win, 0, None)
    b *= sin_win
    return a, b, L, l


def allp_ab(tau):
    L = int(tau) + 1
    l = np.arange(0, L)
    a_0 = np.array([1.0])
    a = np.array(np.cumprod(np.divide(np.multiply((L - l), (L - l - tau)), np.multiply((l + 1), (l + 1 + tau)))))
    a = np.append(a_0, a)  # Denominator of the transfer function
    b = np.flipud(a)  # Numerator of the transfer function
    return a, b, L, l


def unmixing(coeffs, X, **kwargs):
    # Applies an unmixing matrix build from coeffs to a stereo signal X
    # Returns the resulting stereo signal X_del
    # forward mixing: [[1, a0*z^-delay0],[a1*z^-delay1, 1]]*X
    # Coeffs: total 4: First 2: crosswise atten. coeff., 2nd 2: Crosswise delays
    # inverse mixing matrix: 1/(1-a0*z^-delay0*a1*z^-delay1) * [[1, -a0*z^-delay0],[-a1*z^-delay1, 1]] *X
    # For 2x2 matrix inverse see e.g.: https://www.mathsisfun.com/algebra/matrix-inverse.html
    # IIR Filter 1/(1-a0*z^-delay0*a1*z^-delay1) in lfilter:
    # scipy.signal.lfilter([1], [1, zeros(delay0+delay1-1), -a0*a1])

    # print("coeffs unm.=", coeffs)
    X_del = np.zeros(X.shape)
    # Add delays:
    # ToDo: check if abs changes the optimization problem
    a = np.abs(coeffs[0:2])  # allow only positive values for delay and attenuation, for CG optimizer
    taus = np.abs(coeffs[2:4])
    # using all-pass delay filter:
    # a0, b0, h = allp_delay_filter(taus[0])
    # a1, b1, h = allp_delay_filter(taus[1])
    a0, b0, h = sinc_delay_filter(taus[0])
    a1, b1, h = sinc_delay_filter(taus[1])

    # Filter initial state
    if 'filter_state_0' in kwargs:
        filter_state_0 = kwargs['filter_state_0']
    else:
        filter_state_0 = scipy.signal.lfilter_zi(b0, a0)

    # Filter initial state
    if 'filter_state_1' in kwargs:
        filter_state_1 = kwargs['filter_state_1']
    else:
        filter_state_1 = scipy.signal.lfilter_zi(b1, a1)

    xd_1, filter_state_0 = scipy.signal.lfilter(b0, a0, X[:, 1], zi=filter_state_0*X[0, 1])
    xd_0, filter_state_1 = scipy.signal.lfilter(b1, a1, X[:, 0], zi=filter_state_1*X[0, 0])

    X_del[:, 0] = X[:, 0] - a[0] * xd_1
    X_del[:, 1] = X[:, 1] - a[1] * xd_0
    return X_del, filter_state_0, filter_state_1


def lp_filter_and_downsample(X, N, fs=None):
    # down-sampling factor N
    # Define low pass filter, 1/N band
    lp = scipy.signal.remez(63, bands=[0, 0.5 / N, 1.0 / N, 1], desired=[1, 0], weight=[1, 100], Hz=2)
    # Apply low-pass filter
    if fs is None:
        lp_N = np.repeat(scipy.signal.lfilter_zi(lp, 1)[..., np.newaxis], X.shape[1], axis=1)
        X_lp, __fs = scipy.signal.lfilter(lp, 1, X, axis=0, zi=lp_N)
    else:
        X_lp, __fs = scipy.signal.lfilter(lp, 1, X, axis=0, zi=fs)
    # Down-sample
    X_lp = X_lp[::N, :]
    return X_lp, __fs


# The function that performs optimization
def find_coeffs_optimization(X, coeffs_initial, downsample: bool):
    N = 8
    if downsample:
        # Filtering & down-sampling
        X = lp_filter_and_downsample(X, N)

    # Optimize for coeffs
    coeffs_minimized = opt.minimize(cost_n_abs_kl, coeffs_initial, args=(X,), method='CG',
                                    options={'disp': False, 'maxiter': 200})
    coeffs = coeffs_minimized.x.copy()

    if downsample:
        coeffs[2:4] *= N

    return coeffs


def cost_n_abs_kl(coeffs, X, fs0=None, fs1=None):
    coeffs = coeffs.ravel()

    # computes the normalized magnitude of the channels and then applies
    # the Kullback-Leibler divergence
    if fs0 is None or fs1 is None:
        X_prime, __fs0, __fs1 = unmixing(coeffs, X)
    else:
        X_prime, __fs0, __fs1 = unmixing(coeffs, X, filter_state_0=fs0, filter_state_1=fs1)

    X_abs = np.abs(X_prime)
    # normalize to sum()=1, to make it look like a probability:
    X_abs[:, 0] = X_abs[:, 0] / np.sum(X_abs[:, 0])
    X_abs[:, 1] = X_abs[:, 1] / np.sum(X_abs[:, 1])

    # Kullback-Leibler Divergence:
    # print("KL Divergence calculation")
    abs_kl = - np.sum(X_abs[:, 0] * np.log((X_abs[:, 0] + 1e-6) / (X_abs[:, 1] + 1e-6)))

    return abs_kl



















