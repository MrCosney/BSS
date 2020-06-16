import numpy as np
import scipy
import scipy.optimize as opt
import scipy.signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


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
def find_coeffs_optimization(X, coeffs_initial):
    N = 8
    # Filtering & down-sampling
    X = lp_filter_and_downsample(X, N)
    # Optimize for coeffs
    coeffs_minimized = opt.minimize(cost_n_abs_kl, coeffs_initial, args=(X,), method='CG',
                                    options={'disp': True, 'maxiter': 200})
    coeffs = coeffs_minimized.x.copy()
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


def cost_grad_n_abs_kl(coeffs, X):
    """Compute gradient and cost function"""
    coeffs = coeffs.ravel()

    e = 1e-6
    dXr, Xr = dXr_dc(X, coeffs)

    cost = - np.sum(Xr[:, 0] * np.log((Xr[:, 0] + e) / (Xr[:, 1] + e)))

    grad_a0 = np.sum((np.log(Xr[:, 0] + e) + Xr[:, 0]/(Xr[:, 0] + e) - np.log(Xr[:, 1] + e)) * dXr[:, 0])
    grad_a1 = np.sum(- Xr[:, 0]/(Xr[:, 1] + e) * dXr[:, 1])
    grad_tau0 = np.sum((np.log(Xr[:, 0] + e) + Xr[:, 0]/(Xr[:, 0] + e) - np.log(Xr[:, 1] + e)) * dXr[:, 2])
    grad_tau1 = np.sum(- Xr[:, 0]/(Xr[:, 1] + e) * dXr[:, 3])

    grad = np.array([grad_a0, grad_a1, grad_tau0, grad_tau1])

    return cost, grad


def dab_dtau(tau):
    a, b, L, k = allp_ab(tau)
    k = k + 1
    da = np.zeros_like(a)
    da[1::] = - np.cumsum(np.cumprod((L+1)*(L - k + 1)/(k*(k + tau)**2)))
    da[0] = 0
    db = np.flipud(da)
    return da, db, a, b, L


def dh_dtau(L, tau):
    """Derive the derivatives of  impulse responses"""
    # TODO: Проверить перенос строк.
    l = np.arange(L)

    dh = - (np.cos(l - tau)/(l - tau) - np.sin(l - tau)/((l - tau)**2)) * np.sin((np.pi/L) * (l + (L + 1)/2 - tau)) \
         - (np.pi/L) * np.sinc(l - tau) * np.cos((np.pi / L) * (l + (L + 1)/2 - tau))

    return dh


def dh_dtau_v2(L, tau):
    """Derive the derivatives of  impulse responses with Ivans Version"""
    l = np.arange(L)

    dh = - (np.cos(l - tau)/(l - tau) - np.sin(l - tau)/((l - tau)**2)) * np.cos((np.pi/(2*L)) * (l + (1/2) - tau)) \
         + (np.pi/(2*L)) * np.sinc(l - tau) * np.sin( (np.pi/(2*L)) * (l + (1 / 2) - tau))

    return dh


def dxd_dtau_fir(L, x, tau):
    """The filtering of the received audio samples with FIR impulse responses"""

    # Assume x=0 for n - k < 0
    x_p = np.pad(x, (L, 0), 'constant', constant_values=(0, 0))

    # FIR impulse response derivative
    dh = dh_dtau(L, tau)

    # Derivative row vector
    dxd = np.zeros_like(x)

    for n in range(x.shape[1]):
        dxd[n] = np.dot(dh, x_p[(n+L):n:-1])

    return dxd


def dxd_dtau_iir(x, tau):
    """The filtering of the received audio samples with IIR impulse responses"""

    # Filter coefficients and their derivatives
    da, db, a, b, L = dab_dtau(tau)

    # Delayed by filter signal samples
    xd = scipy.signal.lfilter(b, a, x)

    # Assume x=0 for n - k < 0
    x_p = np.pad(x, (L, 0), 'constant', constant_values=(0, 0))
    # Assume xd=0 for n - k < 0
    xd_p = np.pad(xd, (L, 0), 'constant', constant_values=(0, 0))
    # Assume dxd=0 for n - k < 0
    dxd_p = np.zeros_like(x_p)

    for n in range(len(x)):
        dxd_p[L+n] = np.dot(db, np.flipud(x_p[n:n+L+1])) \
                     - np.dot(da[1::], np.flipud(xd_p[n:n+L])) \
                     - np.dot(a[1::], np.flipud(dxd_p[n:n+L]))

    return dxd_p[L::], xd


def dXu_dc(X, coeffs):
    """Unmixing audio data"""
    """Note that a is not a filter coefficients here, 
    but amplitude coefficient of the unmixing matrix"""
    # Signals samples
    x0 = X[:, 0]  # first channel samples
    x1 = X[:, 1]  # second channel samples
    # Coefficients
    a0 = coeffs[0]
    a1 = coeffs[1]
    tau0 = coeffs[2]
    tau1 = coeffs[3]
    # Derivatives of the delayed samples
    dx0d_dtau1, x0d = dxd_dtau_iir(x0, tau1)
    dx1d_dtau0, x1d = dxd_dtau_iir(x1, tau0)
    # Unmixed signals
    x0u = x0 - a0 * x1d
    x1u = x1 - a1 * x0d
    # Derivatives of the unmixed samples
    # First channel
    dx0u_da0 = - x1d
    dx0u_dtau0 = - a0*dx1d_dtau0
    dx0u_da1 = np.zeros_like(dx0u_da0)
    dx0u_dtau1 = np.zeros_like(dx0u_dtau0)
    # Second channel
    dx1u_da1 = - x0d
    dx1u_dtau1 = - a1*dx0d_dtau1
    dx1u_da0 = np.zeros_like(dx1u_da1)
    dx1u_dtau0 = np.zeros_like(dx1u_dtau1)
    # Store it in matrix
    dXu = np.column_stack((dx0u_da0, dx1u_da1, dx0u_dtau0, dx1u_dtau1))
    Xu = np.column_stack((x0u, x1u))

    return dXu, Xu


def dXa_dc(X, coeffs):
    """Final cost function, the absolute value is taken from all unmixed (real-valued) samples"""
    # Compute derivative of the unmixed signal samples
    dXu, Xu = dXu_dc(X, coeffs)
    # Modulus step
    Xa = np.abs(Xu)
    dXa = np.zeros_like(dXu)

    Xu_sign = np.sign(Xu)

    dXa[:, 0] = np.multiply(Xu_sign[:, 0], dXu[:, 0])  # dx0a_da0
    dXa[:, 1] = np.multiply(Xu_sign[:, 1], dXu[:, 1])  # dx1a_da1
    dXa[:, 2] = np.multiply(Xu_sign[:, 0], dXu[:, 2])  # dx0a_dtau0
    dXa[:, 3] = np.multiply(Xu_sign[:, 1], dXu[:, 3])  # dx1a_dtau1

    return dXa, Xa


def dXr_dc(X, coeffs):
    """Normalization of the samples"""
    from math import fsum  # Fsum более точен для цифр с плавающей точкой

    dXa, Xa = dXa_dc(X, coeffs)

    Xr = np.zeros_like(Xa)  # X_norm = np.zeros_like(X_abs)
    dXr = np.zeros_like(dXa)

    # Sums of the absolute values
    x0a_sum = np.sum(Xa[:, 0])
    x1a_sum = np.sum(Xa[:, 1])

    # normalize to sum()=1, to make it look like a probability:
    Xr[:, 0] = Xa[:, 0] / x0a_sum
    Xr[:, 1] = Xa[:, 1] / x1a_sum

    dXr[:, 0] = np.multiply((x0a_sum - Xa[:, 0])/x0a_sum, dXa[:, 0])  # dx0a_da0
    dXr[:, 1] = np.multiply((x1a_sum - Xa[:, 1])/x1a_sum, dXa[:, 1])  # dx1a_da1
    dXr[:, 2] = np.multiply((x0a_sum - Xa[:, 0])/x0a_sum, dXa[:, 2])  # dx0a_dtau0
    dXr[:, 3] = np.multiply((x1a_sum - Xa[:, 1])/x1a_sum, dXa[:, 3])  # dx1a_dtau1

    # TODO: Возможно вытащить из массива fsum(X_abs[i] для упрощения расчетов?
    # for i in range(self.__ch):
    #    X_norm[i] = [z / fsum(X_abs[i]) for z in X_abs[i]]

    # dxr_da = np.zeros_like(X_abs)
    # dxr_dd = np.zeros_like(X_abs)

    # for i in range(self.__ch):
    #     for n in range(X_abs.shape[1]):
    #         dxr_da[i][n] = ((fsum(X_abs[i]) - X_abs[i][n]) / (fsum(X_abs[i]) ** 2)) * dxa_da[i][n]
    #         dxr_dd[i][n] = ((fsum(X_abs[i]) - X_abs[i][n]) / (fsum(X_abs[i]) ** 2)) * dxa_dd[i][n]

    return dXr, Xr


def gradient(self, X_norm, dxr_da, dxr_dd):
    """Gradient calculation"""
    from math import log1p as log
    from math import fsum
    e = 10 ** -6

    gradient = [[], [], [], []]

    for n in range(X_norm.shape[1]):
        gradient[0].append(
            ((log(X_norm[0][n]) + e) + (X_norm[0][n] / (X_norm[0][n] + e)) - (log(X_norm[1][n]) + e)) * dxr_da[0][n])
        gradient[1].append((-(X_norm[0][n] / (X_norm[1][n] + e))) * dxr_da[1][n])
        gradient[2].append(
            ((log(X_norm[0][n]) + e) + (X_norm[0][n] / (X_norm[0][n] + e)) - (log(X_norm[1][n]) + e)) * dxr_dd[0][n])
        gradient[3].append((-(X_norm[0][n] / (X_norm[1][n] + e))) * dxr_dd[1][n])

    [fsum(z) for z in gradient]

    return gradient


def check_gradient():
    # Generate arbitrary data
    X = np.random.randn(1024, 2)
    # Smooth it for nicer picture
    N = 16
    lp = scipy.signal.remez(63, bands=[0, 0.5 / N, 0.7 / N, 1], desired=[1, 0], weight=[1, 100], Hz=2)
    # Apply low-pass filter
    X = scipy.signal.lfilter(lp, [1], X, axis=0)

    import matplotlib.pyplot as plt

    # Point in the parameter space where we check the gradient
    coeffs_at_touch_point = np.array([0.2, 0.7, 3.4, 5.3])
    # direction along which we evaluate cost function & gradient
    coeffs_direction = coeffs_at_touch_point + 0.1
    # We vary theta around touch point
    thetas = np.arange(-1.0, 1.0, 0.001)*0.01
    # Allocate
    cf = np.zeros_like(thetas)
    gr_line = np.zeros_like(thetas)
    # We compute the cost function & gradient at touch point
    cost_at_touch_point, gradient_at_touch_point = cost_grad_n_abs_kl(coeffs_at_touch_point, X)
    eps = np.sqrt(np.finfo(float).eps)
    gradient_at_touch_point = opt.approx_fprime(coeffs_at_touch_point, cost_n_abs_kl, eps, X)
    # Iterate over all thetas, draw cost function & it's 1-st order approximation at touch point
    it = np.nditer(thetas, flags=['f_index'])
    while not it.finished:
        i = it.index
        theta = it.value
        coeffs = coeffs_at_touch_point + theta*(coeffs_at_touch_point - coeffs_direction)
        cf[i] = cost_n_abs_kl(coeffs, X)
        gr_line[i] = cost_at_touch_point - np.dot(gradient_at_touch_point, coeffs_at_touch_point - coeffs)
        it.iternext()

    fig, ax = plt.subplots()
    ax.plot(thetas, cf, 'b-', thetas, gr_line, 'r--')

    ax.set(xlabel='theta', ylabel='Cost function & 1-st order approximation')
    ax.grid()

    plt.show()


def check_separation():
    sample_rate, x = wav.read("wav/stereovoices.wav")

    x = x * 1.0 / np.max(abs(x))

    # Show mixed channels
    x_o = x.copy()
    plt.plot(x_o[:, 0])
    plt.plot(x_o[:, 1])
    plt.title('AIRES.py: The mixed channels')
    plt.show()

    coeffs = find_coeffs_optimization(x, np.array([0.0, 0.0, 0.0, 0.0]))
    print(coeffs)

    x_del, fs0, fs1 = unmixing(coeffs, x_o)

    plt.plot(x_del[:, 0])
    plt.plot(x_del[:, 1])
    plt.title('AIRES.py: The unmixed channels')
    plt.show()


if __name__ == "__main__":
    # check_gradient()
    check_separation()
    # ToDo: check_step1(), check_step2(), etc ...
