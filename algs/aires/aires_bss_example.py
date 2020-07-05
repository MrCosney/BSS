'''
AIRES (time domAIn fRactional dElay Separation)

Joint entropy estimation, mutual information estimation,
using KL-Divergence on magnitude of the time signal,
and optimization for Independent Component Analysis using minimized mutual information,
to obtain directly an unmixing matrix out of the optimization!
Including a delay for the mixing process, with a pure delay.
assumed Mixing matrix:[[1, a0*delay0],[a1*delay1, 1]]
inverse: 1/(1-a0*delay0*a1*delay1) * [[1, -a0*delay0],[-a1*delay1, 1]]
Dropping the inverse of the determinant (IIR filter) improves optimization.
Low pass filtering smoothes away lokal minima in the error function, and hence allows to use
simpler optimization algorithms, like SLSQP, which makes it much faster, and
downsampling the lowpass filtered audio signal makes it again faster!
A pure delay is a somewhat realistic physical model, and since each delay is just 1 variable it is
easier to optimize than FIR filters with many coefficients. Also, a pure delay leads to less sound coloration
after unmixing than a long FIR filter might.


Gerald Schuller, Oleg Golokolenko, March 2019
'''



import numpy as np
import time
import scipy.signal
from scipy import io


def playsound(audio, samplingRate, channels):
    # funtion to play back an audio signal, in array "audio"
    import pyaudio
    p = pyaudio.PyAudio()
    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)

    audio = np.clip(audio, -2 ** 15, 2 ** 15 - 1)
    sound = (audio.astype(np.int16).tostring())
    # sound = (audio.astype(np.int16).tostring())ICAabskl_puredelay_lowpass.py
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return


def allp_delayfilt(tau):
    '''
    performs Fractional-delay All-pass Filter
    :param tau: fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
    :type tau: float or int
    :return:
        a: Denumerator of the transfer function
        b: Numerator of the transfer function
    '''

    import scipy.signal
    # print("tau", tau)
    # Pay attention to "int()" function. It doesn't nor rounds no floors the value. Just removes decimal part
    L = int(tau) + 1
    n = np.arange(0, L)

    a_0 = np.array([1.0])
    a = np.array(np.cumprod(np.divide(np.multiply((L - n), (L - n - tau)), (np.multiply((n + 1), (n + 1 + tau))))))
    a = np.append(a_0, a)  # Denumerator of the transfer function
    # print("Denumerator of the transfer function a:", a)

    b = np.flipud(a)  # Numerator of the transfer function
    # print("Numerator of the transfer function b:", b)

    # Calculate Impulse response of the filter (Not for implementation on Android)
    # Nh = L * 1  # Length of the transfer function
    # impulse_train = np.zeros(Nh + 1)
    # impulse_train[0] = 1
    # h = scipy.signal.lfilter(b, a, impulse_train)  # impulse response

    return a, b

def unmixing(coeffs, X):
    '''
    Applies an unmixing matrix build from coeffs to a stereo signal X
    Returns the resulting stereo signal X_del
    forward mixing: [[1, a0*z^-delay0],[a1*z^-delay1, 1]]*X
    Coeffs: total 4: First 2: crosswise atten. coeff., 2nd 2: Crosswise delays
    inverse mixing matrix: 1/(1-a0*z^-delay0*a1*z^-delay1) * [[1, -a0*z^-delay0],[-a1*z^-delay1, 1]] *X
    For 2x2 matrix inverse see e.g.: https://www.mathsisfun.com/algebra/matrix-inverse.html
    IIR Filter 1/(1-a0*z^-delay0*a1*z^-delay1) in lfilter:
    scipy.signal.lfilter([1], [1, zeros(delay0+delay1-1), -a0*a1])

    :param coeffs: Optimization parameters
    :type coeffs: 1d np.array
    :param X: Signal to be separated
    :type X: 2d np.array of size [M_samples, K_channels]. Here, only K=2 is implemented
    :return: Separated stereo signal
    :rtype: 2d np.array of size [M_samples, K_channels]. Here, only K=2 is implemented
    '''

    X_del = np.zeros(X.shape)
    # Add delays:
    a = np.abs(coeffs[0:2])  # allow only positive values for delay and attenuation
    delay = np.abs(coeffs[2:4])

    # Calculate delay filter coefficients
    a0, b0 = allp_delayfilt(delay[0])
    a1, b1 = allp_delayfilt(delay[1])

    X_del[:, 0] = X[:, 0] - a[0] * scipy.signal.lfilter(b0, a0, X[:, 1])
    X_del[:, 1] = X[:, 1] - a[1] * scipy.signal.lfilter(b1, a1, X[:, 0])
    return X_del

def minabsklcoeffs(coeffs, X):
    '''
    Computes the normalized magnitude of the channels and then applies the Kullback-Leibler divergence
    :param coeffs: Optimization parameters
    :type coeffs: 1d np.array
    :param X: Signal to be separated
    :type X: 2d np.array of size [M_samples, K_channels]. Here, only K=2 is implemented
    :return: Error/Cost function value
    :rtype: float
    '''

    X_minabskl = np.copy(X)  # Just in case. To be sure that original variable X is not changed in current function

    X_prime = unmixing(coeffs, X_minabskl)
    X_prime_abs = np.abs(X_prime)
    # normalize to sum()=1, to make it look like a probability:
    X_prime_abs[:, 0] = X_prime_abs[:, 0] / np.sum(X_prime_abs[:, 0])
    X_prime_abs[:, 1] = X_prime_abs[:, 1] / np.sum(X_prime_abs[:, 1])

    abskl1 = np.sum(X_prime_abs[:, 0] * np.log((X_prime_abs[:, 0] + 1e-6) / (X_prime_abs[:, 1] + 1e-6)))
    abskl2 = np.sum(X_prime_abs[:, 1] * np.log((X_prime_abs[:, 1] + 1e-6) / (X_prime_abs[:, 0] + 1e-6)))
    abskl = (abskl1 + abskl2)

    return -1.0*abskl


def aires_separation_offline(X, maxdelay = 20, max_atten = 1, opt_it = 30, coeffs = np.array([1.0, 1.0, 1.0, 1.0]), coeffweights_orig = np.array([0.1, 0.1, 1.0, 1.0]), betha = 0.8, do_downsampling = False, N_downsample = 2, show_progress = False):
    '''
    Function performs offline AIRES BSS

    :param X: 2 channel mixed audio
    :type X: 2d np.array of size [M_samples, K_channels]. Here, only K=2 is implemented

    :param maxdelay: Maximum delay that is used for separation
    :type maxdelay: int

    :param max_atten: Minimum/Maximum attenuation factors that are used for separation
    :type max_atten: int

    :param opt_it: Number of optimization/separation iterations
    :type opt_it: int

    :param coeffs: Starting separation coefficients
    :type coeffs: 1d np.array of floats. Length = 4. Where coeffs[0:2] - attenuation factors; coeffs[2:4] - delays

    :param coeffweights_orig: Coefficients weights
    :type coeffweights_orig: 1d np.array of floats. Length = 4

    :param betha: Factor for coeffweights_orig
    :type betha: float

    :param do_downsampling: Activate/deactivate low pass signal filtering and Downsample
    :type do_downsampling: True/False

    :param N_downsample: Downsampling ration (how many samples we keep from original signal). Only for optimization routing
    :type N_downsample: int

    :param show_progress: Show or not optimized cost function value  (can be removed for Android)
    :type show_progress: True/False (can be removed for Android)

    :return: 2 channel unmixed audio
    :rtype: 2d np.array of size [M_samples, K_channels]. Here, only K=2 is implemented
    '''

    X_mixed_orig = np.copy(X)   # Just in case. To be sure that original variable X is not changed in current function

    if do_downsampling == True:
        # Downsample mixed signal and low-pass it to make cost function more convex, simplify and make convolution faster
        # Create Low-pass filter, 1/N band, before downsamplin:
        lp = scipy.signal.remez(63, bands=[0, 0.32 / N_downsample, 0.45 / N_downsample, 1], desired=[1, 0], weight=[1, 100], Hz=2)
        # Filter signal
        X_mixed_lp = scipy.signal.lfilter(lp, [1], X_mixed_orig, axis=0)
        # Downsample signal
        X_mixed = X_mixed_lp[::N_downsample, :]
    else:
        X_mixed = X_mixed_orig.copy()


    """#######################################################################################"""
    # Starting point for optimization / Starting cost function value
    negabskl_0 = minabsklcoeffs(coeffs, X_mixed)
    coeffweights = coeffweights_orig * betha

    for m in range(opt_it):
        # Create 1d vector of random variables between -0.5 and +0.5
        # Pay attention at this point, since different random function can give different random number distributions
        # Here, uniform distribution is implemented
        rnd_variation = (np.random.rand(4) - 0.5)
        coeffvariation = rnd_variation * coeffweights # small variation
        coeffs_tmp = (coeffs + coeffvariation)

        # Clip/cut the variables according to their max and min values
        coeffs_tmp[0:2] = np.clip(coeffs_tmp[0:2], -1.0*max_atten, max_atten)  # limit possible attenuation factors
        coeffs_tmp[2:4] = np.clip(coeffs_tmp[2:4], 0, maxdelay)  # allow only range of 0 to maxdelay

        # Calculate new cost function value
        negabskl = minabsklcoeffs(coeffs_tmp, X_mixed)

        if negabskl < negabskl_0:  # if variation was successful, keep that point  "New is better"
            negabskl_0 = negabskl
            coeffs = coeffs_tmp
            if show_progress == True:   # can be removed for Android (Up to you)
                print(m, negabskl)



    """#######################################################################################"""
    if do_downsampling == True:
        coeffs[2:4] *= N_downsample  # multiply the delays according to the downsampling factor

    X_unmixed = unmixing(coeffs, X_mixed_orig)

    return X_unmixed



if __name__ == '__main__':

    # ToDo: Variables that have to be set up in the application (maybe as "Settings" or additional tab.
    # To not show on the main screen). They are placed here in order of importance
    samplerate = 16000  # in Hz
    maxdelay  = 20  # in Samples
    max_atten = 2
    opt_it = 30
    coeffs = np.array([1.0, 1.0, 1.0, 1.0])
    coeffweights_orig = np.array([0.1, 0.1, 1.0, 1.0])
    betha = 0.8
    do_downsampling = False
    N_downsample = 2
    show_progress = False
    play_volume = 1.0 # float
    playing_cut_time = 3  # in seconds


    '''###########################################################'''
    """ Open simulated convolutive mixture """
    # ToDo: Make it possible to choose the file you use for separation
    f_open_name = "stationary_ss_rt60-0.05_TIMIT_dist-2_mix-0-0.mat"
    x_load = scipy.io.loadmat(f_open_name)
    X_mixed = x_load['mixed_ss']        # Mixed channels
    # ToDo: For android you can save this variable "X_mixed" to a file that you can open in Android

    # Normalize signal  (abs(Max_value) has to be 1)
    X_mixed = X_mixed/np.max(np.abs(X_mixed))

    # ToDo: Sometimes (such in current case) you dont wanna listen to entire music file. Limit it to some time for listening
    cut_time = playing_cut_time * samplerate
    if cut_time > len(X_mixed[:, 0]):
        cut_time = len(X_mixed[:, 0])

    # Play back Mixed channels  (As a button + possibility to choose the channel number)
    # Do not forget to return the signals original amplitude
    # X_mixed = play_volume * X_mixed * 2 ** 15
    # play_channel_n = 0
    # playsound(X_mixed[:cut_time, play_channel_n], samplerate, 1)
    # play_channel_n = 1
    # playsound(X_mixed[:cut_time, play_channel_n], samplerate, 1)

    '''###########################################################'''
    # Apply separation
    X_unmixed = aires_separation_offline(X_mixed, maxdelay, max_atten, opt_it, coeffs, coeffweights_orig, betha, do_downsampling, N_downsample, show_progress)
    # ToDo: Save separated audio
    '''###########################################################'''

    # Play back separated channels (As a button + possibility to choose the channel number)
    # Do not forget to return the signals original amplitude
    X_unmixed = play_volume * X_unmixed * 2 ** 15
    play_channel_n = 0
    playsound(X_unmixed[:cut_time, play_channel_n], samplerate, 1)
    play_channel_n = 1
    playsound(X_unmixed[:cut_time, play_channel_n], samplerate, 1)