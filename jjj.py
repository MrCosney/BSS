# Joint entropy estimation, mutual information estimation,
# and optimization for Independent Component Analysis using minimized mutual information,
# to obtain directly an unmixing matrix out of the optimization!
# Including a delay for the mising process, with a pure delay.
# assumed Mixing matrix:[[1, a0*delay0],[a1*delay1, 1]]
# inverse: 1/(1-a0*delay0*a1*delay1) * [[1, -a0*delay0],[-a1*delay1, 1]]
# Gerald Schuller, October 2017

import numpy as np


def frac_delay_filt(tau):
    '''
    performs Fractional-delay All-pass Filter
    :param tau: fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
    :type tau: float or int
    :return:
        a: Denumerator of the transfer function
        b: Numerator of the transfer function
        h: Transfer function of the filter
    '''

    import scipy
    import scipy.signal as sp

    L = max(1, int(tau))
    n = np.arange(0, L)
    # print("n", n)

    a_0 = np.array([1.0])
    a = np.array(np.cumprod(np.divide(np.multiply((L - n), (L - n - tau)), (np.multiply((n + 1), (n + 1 + tau))))))
    a = np.append(a_0, a)  # Denumerator of the transfer function
    # print("Denumerator of the transfer function a:", a)

    b = np.flipud(a)  # Numerator of the transfer function
    # print("Numerator of the transfer function b:", b)

    # Calculate Impulse response of the filter
    Nh = L * 5  # Length of the transfer function
    impulse_train = np.zeros(Nh + 1)
    impulse_train[0] = 1
    h = scipy.signal.lfilter(b, a, impulse_train)  # Transfer function

    return a, b, h


def entropy(x, bins):
    # Function to extimate the entropy of a signal (in a 1d array)
    # arguments: x: 1d array contains a signal
    # bins: # of bins for the histogram computation
    # returns:  entropy in terms of natural logarithm (nats)

    Nanindex = np.argwhere(np.isnan(x))
    if any(Nanindex):
        print("Not a number error, index=", Nanindex)
    hist, binedges = np.histogram(x, bins)
    # print("hist=", hist)
    pdf = 1.0 * hist / np.sum(hist)
    entr = - np.sum(pdf * np.log(pdf + 1e-6))
    return entr


def entropy2d(x, bins):
    # Function to extimate the joint entropy of 2 signals (in a 2d array)
    # arguments: x: 2d array, each of its 2 columns contains a signal
    # bins: # of bins for the histogram computation
    # returns: joint entropy in terms of natural logarithm (nats)

    hist2d, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], bins)
    # print("hist2d=",hist2d)
    # print("sum=", np.sum(hist2d))
    pdf2d = 1.0 * hist2d / np.sum(hist2d)
    # print("pdf2d=",pdf2d)
    # joint entropy according to: https://en.wikipedia.org/wiki/Joint_entropy
    jointentropy = - np.sum(pdf2d * np.log(pdf2d + 1e-6))
    return jointentropy


def mutualinformation(x, bins):
    # Computes the mutual information I(X;Y) of 2 signals in x
    # Arguments:
    # x: 2d array with 2 columns, 1 column for each signal
    # returns: mutual information absed on the natural logarithm (nats)
    # Computation based on https://en.wikipedia.org/wiki/Mutual_information
    # I(X;Y)= H(X)+H(Y)-H(X,Y)
    # from matplotlib.mlab import entropy

    HX = entropy(x[:, 0], bins)  # entropy of x

    HY = entropy(x[:, 1], bins)  # entropy of Y
    # print("HX=", HX, "HY=", HY)
    HXY = entropy2d(x, bins)  # joint entropy of X and Y
    return HX + HY - HXY


def delayfilt(delay):
    # FIR filter for applying a fractional delay "delay" to a filter
    # Returns the FIR filter "delayfilt"
    # usage: X_del=lfilter(delayfilt(delay),[1],X)
    delayfilt = np.sinc(np.arange(100) - delay)
    # windowing:
    sinwin = np.sin(np.pi / 100 * (np.arange(100) + 0.5 + 50 - delay))
    sinwin = np.clip(sinwin, 0, None)
    # print("sinwin=", sinwin)
    delayfilt *= sinwin
    return delayfilt


def unmixing(coeffs, X):
    # Applies an anmixing matrix build from coeffs to a stereo signal X
    # Returns the resulting stereo signal X_del
    # forward mixing: [[1, a0*z^-delay0],[a1*z^-delay1, 1]]*X
    # Coeffs: total 4: First 2: crosswise Unmixing coeff., 2nd 2: Crosswise delays
    # inverse mixing matrix: 1/(1-a0*z^-delay0*a1*z^-delay1) * [[1, -a0*z^-delay0],[-a1*z^-delay1, 1]] *X
    # For 2x2 matrix inverse see e.g.: https://www.mathsisfun.com/algebra/matrix-inverse.html
    # IIR Filter 1/(1-a0*z^-delay0*a1*z^-delay1) in lfilter:
    # scipy.signal.lfilter([1], [1, zeros(delay0+delay1-1), -a0*a1])

    X_del = np.zeros(X.shape)
    # Add delays:
    a = coeffs[0:2]
    delay = coeffs[2:4]
    # print("delay=", delay)

    # '''
    # delay filter of length 100, with sinc main lobe in the center:
    # delayfilt=np.sinc(np.arange(100)-50)
    # windowing:
    # sinwin=np.sin(np.pi/100*(np.arange(100)+0.5))
    # delayfilt*=sinwin
    # delay filters for fractional delays:
    # delayfilt0=np.sinc(np.arange(30)-delay[0])#*sinwin
    # delayfilt0=delayfilt(delay[0])
    # #delayfilt1=np.sinc(np.arange(30)-delay[1])#*sinwin
    # delayfilt1=delayfilt(delay[1])
    # X_del[:,0]=X[:,0]-a[0]*scipy.signal.lfilter(delayfilt0, 1,X[:,1])
    # X_del[:,1]=X[:,1]-a[1]*scipy.signal.lfilter(delayfilt1, 1,X[:,0])
    # '''

    # '''
    #################################################
    # Added by Oleg

    a0, b0, h0 = frac_delay_filt(delay[0])
    a1, b1, h1 = frac_delay_filt(delay[1])

    X_del[:, 0] = X[:, 0] - a[0] * scipy.signal.lfilter(b0, a0, X[:, 1])
    X_del[:, 1] = X[:, 1] - a[1] * scipy.signal.lfilter(b1, a1, X[:, 0])
    #################################################
    # '''

    # make IIR filter 1/(1-a0*delay0*a1*delay1):
    # totaldelfilt=-np.sinc(np.arange(60)-delay[0]-delay[1])
    # print("np.arange(60)-delay[0]-delay[1]=",np.arange(60)-delay[0]-delay[1])
    # print("Totaldelayfilt=", totaldelfilt)
    # totaldelfilt= totaldelfilt *a[0]*a[1]
    # totaldelfilt[0]+= 1
    # print("Totaldelayfilt=", totaldelfilt)
    # X_del[:,0]=scipy.signal.lfilter([1], totaldelfilt, X_del[:,0])
    # X_del[:,1]=scipy.signal.lfilter([1], totaldelfilt, X_del[:,1])
    return X_del


def mutualinfoangles(angles, X):
    # computes the mutual information of 2 signals in array X columns, after multiplying with
    # an "unmixing" matrix A
    # to find the unmixing matrix which produces the minimum mutual information

    # print("angles= ", angles)
    alpha = angles[0]
    beta = angles[1]
    A = np.array([[np.cos(alpha), np.sin(alpha)], [np.cos(beta), np.sin(beta)]]).T
    X_prime = np.dot(X, A)
    mutualinfo = mutualinformation(X_prime, 100)
    # print("mutualinfo=", mutualinfo)
    return mutualinfo


def mutualinfocoeffs(coeffs, X):
    # computes the mutual information of 2 signals in array X columns, with pure delays
    # to find the unmixing coefficients which produces the minimum mutual information

    # print("coeffs= ", coeffs)
    X_del = unmixing(coeffs, X)
    mutualinfo = mutualinformation(X_del, 100)
    # print("mutualinfo=", mutualinfo)
    return mutualinfo


def minfftklcoeffs(coeffs, X):
    # computes the normalized magnitude FFT of the channels and then applies
    # the Kullback-Leibler divergence
    X_prime = unmixing(coeffs, X)
    X_fft = np.abs(np.fft.rfft(X_prime, axis=0))
    # normalize to sum()=1, to make it look like a probability:
    X_fft[:, 0] = X_fft[:, 0] / np.sum(X_fft[:, 0])
    X_fft[:, 1] = X_fft[:, 1] / np.sum(X_fft[:, 1])
    # print("coeffs=", coeffs)
    # Kullback-Leibler Divergence:
    fftkl = np.sum(X_fft[:, 0] * np.log((X_fft[:, 0] + 1e-6) / (X_fft[:, 1] + 1e-6)))
    # print("fftkl=", fftkl)
    return -fftkl


def minabsklcoeffs(coeffs, X):
    # computes the normalized magnitude of the channels and then applies
    # the Kullback-Leibler divergence
    X_prime = unmixing(coeffs, X)
    X_abs = np.abs(X_prime)
    # normalize to sum()=1, to make it look like a probability:
    X_abs[:, 0] = X_abs[:, 0] / np.sum(X_abs[:, 0])
    X_abs[:, 1] = X_abs[:, 1] / np.sum(X_abs[:, 1])

    # print("coeffs=", coeffs)
    # Kullback-Leibler Divergence:
    # print("KL Divergence calculation")
    abskl = np.sum(X_abs[:, 0] * np.log((X_abs[:, 0] + 1e-6) / (X_abs[:, 1] + 1e-6)))
    return -abskl


def itakurasaitocoeffs(coeffs, X):
    # computes the normalized magnitude of the channels and then applies
    # the Itakura-Saito Distance
    X_prime = unmixing(coeffs, X)
    X_abs = np.abs(X_prime)
    # normalize to sum()=1, to make it look like a probability:
    X_abs[:, 0] = X_abs[:, 0] / np.sum(X_abs[:, 0])
    X_abs[:, 1] = X_abs[:, 1] / np.sum(X_abs[:, 1])

    # print("coeffs=", coeffs)
    # Itakura-Saito Distance:
    # print("Itakura-Saito Distance")
    itakurasaito = np.sum(
        X_abs[:, 0] / (X_abs[:, 1] + 1e-6) - np.log((X_abs[:, 0] + 1e-6) / (X_abs[:, 1] + 1e-6)) - 1) / 100000
    return -itakurasaito


def meanpower(coeffs, X):
    # computes the power of the umnixed channels
    # For power minimization like the LMS algorithm
    X_prime = unmixing(coeffs, X)
    power = np.sum(X_prime ** 2)
    return power


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
    # sound = (audio.astype(np.int16).tostring())
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return


# testing:
if __name__ == '__main__':
    import scipy.io.wavfile as wav
    import scipy.optimize as opt
    import scipy.signal
    import os
    import matplotlib.pyplot as plt

    # print("Testing delay filter, delayfilt(4.5)=", delayfilt(4.5))
    print("Testing entropy function:")
    print("Entropy()=", entropy(np.array([0, 1.0, 2.0]), 20))

    samplerate, sources = wav.read("stereovoices.wav")
    sources = sources * 1.0 / 2 ** 15
    # To avoid overflow reduce amplitude:
    sources *= 0.5
    print("type(sources)=", type(sources))
    jointentropy = entropy2d(sources, 100)
    print("Joint Entropy of unmixed signals =", jointentropy)

    mutualinfo = mutualinformation(sources, 100)
    print("Mutual Information of unmixed signals =", mutualinfo)

    abskl = minabsklcoeffs(np.zeros(4), sources)
    print("Kullback-Leibler of abs of time signals of unmixed signals=", abskl)
    # Mixing:
    """
    a=0.45
    b = 1 - a
    A= np.array([[a, b], [b, a]])
    X=np.dot(sources,A)
    """
    X = np.zeros(sources.shape)
    # Do the mixing as in a stereo microphone pairs. Add delayed and attenuated 2nd channel to each:
    # attenuation: 0.9 for each, delay of 4 samples for each:

    coeffs = np.array([0.9, 0.9, 5, 5])
    # shifted sinc functions over 10 samples to approximate fractional delays:
    # forward mixing: [[1, 0.9*z^-4],[0.9*z^-4, 1]]*X
    sinwin = np.sin(np.pi / 30 * (np.arange(30) + 0.5))
    delayfilt0 = delayfilt(coeffs[2])
    # delayfilt1=np.sinc(np.arange(30)-coeffs[3])#*sinwin
    delayfilt1 = delayfilt(coeffs[3])
    X[:, 0] = 1.0 * sources[:, 0] + coeffs[0] * scipy.signal.lfilter(delayfilt0, [1], sources[:, 1])
    X[:, 1] = 1.0 * sources[:, 1] + coeffs[1] * scipy.signal.lfilter(delayfilt1, [1], sources[:, 0])

    N = 8  # downsampling by N, can be changed

    lp = scipy.signal.remez(63, bands=[0, 0.32 / N, 0.45 / N, 1], desired=[1, 0], weight=[1, 100], Hz=2)
    w, H = scipy.signal.freqz(lp)
    plt.plot(w, 20 * np.log10(np.abs(H)))
    plt.show()
    Xlp = scipy.signal.lfilter(lp, [1], X, axis=0)
    Xlp = Xlp[::N, :]
    Xorig = X.copy()
    X = Xlp

    plt.plot(X[:, 0])
    plt.plot(X[:, 1])
    plt.title('The mixed channels')
    plt.show()
    # 'coeffs_minimized', array([ 0.97112147, -0.95145071,  1.97111766,  2.37623401]))
    # 'coeffs_minimized', array([  0.99531339,  -0.79257013,   9.11810359,  13.27731099]
    # 'coeffs_minimized', array([  0.99,  -0.98,   9.08977264,  13.64741446])
    # 'coeffs_minimized', array([  0.9089493 ,   0.12405548,   9.70591602,  10.46540297])
    # 'coeffs_minimized', array([  0.75783302,   0.3285148 ,   8.65370068,  11.12170656])

    os.system('espeak -s 120 "The mixed signal, stereo"')
    # playsound(Xorig*2**15, samplerate, 2)

    """
    #an optim run: coeffs=np.array([ 0.97085815,  0.94538317,  4.73053544,  3.6983346 ])
    #Since we know the mixing coefficients, we can test the correct ideal unmixing matrix:
    X_del=unmixing(coeffs,X)
    os.system('espeak -s 120 "Testing the unmixed signal with correct coefficients"')
    os.system('espeak -s 120 "Channel 0"')
    playsound(X_del[:,0]*2**15, samplerate, 1)
    os.system('espeak -s 120 "Channel 1"')
    playsound(X_del[:,1]*2**15, samplerate, 1)
    """
    # """
    # abskl=np.zeros(200)
    # meanpow=np.zeros(200)
    # itakurasaito=np.zeros(200)
    # mutualinfo=np.zeros(200)
    # for delay in range(0,200):
    #   abskl[delay]=minabsklcoeffs(np.array([0.9,0.9,5.0/N,delay*0.1/N]),X)
    #   #meanpow[delay]=meanpower(np.array([0.9,0.9,5.0,delay]),X)
    #   itakurasaito[delay]=itakurasaitocoeffs(np.array([0.9,0.9,5.0/N,delay*0.1/N]),X)
    #   mutualinfo[delay]=mutualinfocoeffs(np.array([0.9,0.9,5.0/N,delay*0.1/N]),X)
    #   #print("minabsklcoeffs([0.9,0.9,5.0,"+str(delay)+"])=",minabsklcoeffs(np.array([0.9,0.9,5.0,delay]),X))
    #   #print("meanpower([0.9,0.9,5.0,"+str(delay)+"])=",meanpower(np.array([0.9,0.9,5.0,delay]),X))
    # plt.plot(np.arange(0,20,0.1),abskl)
    # #plt.plot(np.arange(0,10,0.1),meanpow/10000)
    # plt.plot(np.arange(0,20,0.1),itakurasaito)
    # plt.plot(np.arange(0,20,0.1),mutualinfo)
    # #plt.legend(('abskl','meanpow/10000'))
    # plt.legend(('abskl','itakurasaito','mutualinfo'))
    # plt.xlabel("Delay in samples")
    # plt.title("Error functions over Delay")
    # plt.show()

    """
    print("Test unmixing:")
    X_del=unmixing(np.array([0.9,0.9,5.0,5.0]),X)
  
    plt.plot(X[:,0])
    plt.plot(X[:,1])
    plt.plot(X_del[:,0])
    plt.plot(X_del[:,1])
    plt.legend(['X0','X1','X_del0','Xdel1'])
    plt.title('Testing unmixing channels')
    plt.show()
  
    os.system('espeak -s 120 "Test Separated Channel 0"')
    playsound(X_del[:,0]*2**15, samplerate, 1)
    os.system('espeak -s 120 "Test Separated Channel 1"')
    playsound(X_del[:,1]*2**15, samplerate, 1)
    """
    # [0.0, np.pi/2]
    # np.pi*np.random.rand(2)
    # coeffs_minimized = opt.minimize(mutualinfocoeffs,[0.9, 0.9, 4.0,4.0]  , args=(X,), method='CG')
    # coeffs_minimized = opt.minimize(minfftklcoeffs,[0.9, 0.9, 4.0,4.0]  , args=(X,), method='CG')
    # angles_minimized = opt.fminbound(mutualinfoangles, np.array([0.0, 0.0]), np.array([np.pi,np.pi]), args=(X,), xtol=1e-05, maxfun=500)
    # """
    # bounds: range of: attenuation 0, attenuation 1, delay 0, delay 1:
    # max magnitude of attenuation coeffs needs to be a little less than 1 for stability of the IIR filter in
    # unmixing, to avoid error message 'range parameter must be finite.':
    # Bound a little lower than 1 to avoid unstability even though a finite sinc approximatio
    # is used:
    bounds = [(0.0 + 1e-4, 1.0 - 1e-4), (0.0 + 1e-4, 1.0 - 1e-4), (0, 15.0), (0, 15.0)]
    # bounds=[(-2.0,2.0)]*4
    # coeffs_minimized  = opt.differential_evolution(mutualinfocoeffs, bounds, args=(X,),tol=1e-4, disp=True )
    # coeffs_minimized  = opt.differential_evolution(minfftklcoeffs, bounds, args=(X,),tol=1e-4, disp=True )
    # about 3 times fastr that fftkl (ca. 35s vs 1m35s):
    # coeffs_minimized  = opt.differential_evolution(minabsklcoeffs, bounds, args=(X,),tol=1e-4, disp=True )
    coeffs_minimized = opt.minimize(minabsklcoeffs, [0.0, 0.0, 0.0, 0.0], args=(X,), bounds=bounds, method='SLSQP',
                                    options={'disp': True, 'maxiter': 200})
    # coeffs_minimized = opt.minimize(minabsklcoeffs, [0.0, 0.0, 0.0, 0.0], args=(X,), method='CG',options={'disp':True, 'maxiter': 200})
    # coeffs_minimized = opt.minimize(mutualinfocoeffs, [0.9, 0.9, 5.0, 5.0], args=(X,), bounds=bounds, method='SLSQP',options={'disp':True, 'maxiter': 200})
    # coeffs_minimized = opt.minimize(meanpower, [0.9, 0.9, 5.0, 5.0], args=(X,), bounds=bounds, method='SLSQP',options={'disp':True, 'maxiter': 200})
    # coeffs_minimized = opt.minimize(itakurasaitocoeffs, [0.9, 0.9, 10.0, 10.0], args=(X,), bounds=bounds, method='SLSQP',options={'disp':True, 'maxiter': 200})
    # coeffs_minimized = opt.minimize(minfftklcoeffs, [0.0, 0.0, 0.0, 0.0], args=(X,), bounds=bounds, method='SLSQP',options={'disp':True, 'maxiter': 200})
    # coeffs_minimized = opt.minimize(mutualinfocoeffs, [0.0, 0.0, 0.0, 0.0], args=(X,), bounds=bounds, method='SLSQP',options={'disp':True, 'maxiter': 200})
    # x: array([ 0.88716832,  0.90042597,  3.97567866,  3.99646177])
    # coeffs_minimized = opt.minimize(minfftklcoeffs, [0.99, -0.98, 9.08977264, 13.64741446], args=(X,), method='CG',options={'disp':True})
    # coeffs_minimized = opt.minimize(minfftklcoeffs, [0.9, 0.9, 5.0, 5.0], args=(X,), method='CG',options={'disp':True})
    # """

    print("angles_minimized=", coeffs_minimized)
    mutualinfo = mutualinfoangles(coeffs_minimized.x, X)
    print("mutualinfo=", mutualinfo)

    coeffs = coeffs_minimized.x
    coeffs[2:4] *= N  # multiply the delays according to the downsampling factor
    print("coeffs_minimized", coeffs)

    X_del = unmixing(coeffs, Xorig)

    plt.plot(X_del[:, 0])
    plt.plot(X_del[:, 1])
    plt.title('The unmixed channels')
    plt.show()

    X_del = X_del * 1.0 / np.max(abs(X_del))
    os.system('espeak -s 120 "Separated Channel 0"')
    playsound(X_del[:, 0] * 2 ** 15, samplerate, 1)
    os.system('espeak -s 120 "Separated Channel 1"')
    playsound(X_del[:, 1] * 2 ** 15, samplerate, 1)
