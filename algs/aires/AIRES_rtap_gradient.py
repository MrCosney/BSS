from algs.aires.AIRES_rtap import *


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
