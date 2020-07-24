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
import pickle
import time
import scipy.signal
import matplotlib.pyplot as plt
import time
from random import randrange, uniform
import mir_eval.separation as mir_eval_separation
import scipy.io.wavfile as wav
import os

class aires_offline_class():

    def __init__(self, coeffs=[1.0, 1.0, 1.0, 1.0], alpha = 0.8, coeffweights = None, n_iter=20, maxdelay = 20.0, maxatten = 2.0, do_downsampling_lowpass = False):
        # Initial coefficients for separation
        self.coeffs = coeffs
        # Number iterations per block
        self.n_iter = n_iter
        # Switch on/off Lowpass filtering and Downsampling
        self.do_downsampling_lowpass = do_downsampling_lowpass
        # Max value for delay
        self.maxdelay = maxdelay
        # Max value for attenuation
        self.maxatten = maxatten
        # Search stepsize
        self.alpha = alpha
        # Small variation for optimization
        self.coeffweights = np.array([0.1, 0.1, 1.0, 1.0]) * alpha


    def aires_offline_separation(self, X):
        self.coeffs = [1.0, 1.0, 1.0, 1.0]
        X_mixed_4_separation = np.copy(X)
        X_mixed_orig = X_mixed_4_separation.copy()


        if self.do_downsampling_lowpass == True:
            # Downsample mixed signal and low-pass it to make cost function more convex to focuse on lowpass part
            # of the signall which has bigger values, simplify and make convolution faster
            N = 2  # downsampling by N, can be changed
            # low pass filter, 1/N band, before downsamplin:
            lp = scipy.signal.remez(63, bands=[0, 0.32 / N, 0.45 / N, 1], desired=[1, 0], weight=[1, 100], Hz=2)
            X_mixed_lp = scipy.signal.lfilter(lp, [1], X_mixed_4_separation, axis=0)
            X_mixed_lp_ds = X_mixed_lp[::N, :]
            X_mixed_4_separation = np.copy(X_mixed_lp_ds)

        negabskl0 = self.minabsklcoeffs(self.coeffs, X_mixed_4_separation)  # Starting point

        for m in range(self.n_iter):
            # if m > (self.n_iter_pro_block):
            #     self.alpha = 0.3
            # np.array([0.07, 0.07, 0.7, 0.7]) * 0.4
            rnd = (np.random.rand(4) - 0.5)
            coeffvariation = rnd * np.array([0.1, 0.1, 1.0, 1.0]) * self.alpha #self.coeffweights  # small variation
            coeffs_tmp = (self.coeffs + coeffvariation)
            coeffs_tmp[0:2] = np.clip(coeffs_tmp[0:2], -1*self.maxatten, self.maxatten)  # limit possible attenuations
            coeffs_tmp[2:4] = np.clip(coeffs_tmp[2:4], -self.maxdelay, self.maxdelay)  # allow only range of 0 to maxdelay
            negabskl1 = self.minabsklcoeffs(coeffs_tmp, X_mixed_4_separation)

            if negabskl1 < negabskl0:  # if variation was successful, keep that point  "New is better"
                negabskl0 = negabskl1
                self.coeffs = coeffs_tmp  # coeffs + coeffvariation
                # if show_progress == True:
            # print(m, negabskl0)

        # print("1 Separation coeffs (atten0, atten1, delay0, delay1): ", self.coeffs)
        if self.do_downsampling_lowpass == True:
            self.coeffs[2:4] = self.coeffs[2:4]*N  # multiply the delays according to the downsampling factor

        # print("Separation process has been done")
        # print("\n *********************************** \n")
        # print("2 Separation coeffs (atten0, atten1, delay0, delay1): ", self.coeffs)
        # print("\n *********************************** \n")

        X_unmixed = self.unmixing(self.coeffs, X_mixed_orig)

        return X_unmixed


    def minabsklcoeffs(self, coeffs, X):
        # Computes the normalized magnitude of the channels and then applies
        # the Kullback-Leibler divergence

        X_minabskl = np.copy(X)

        X_prime = self.unmixing(coeffs, X_minabskl)
        X_prime_abs = np.abs(X_prime)
        # normalize to sum()=1, to make it look like a probability:
        X_prime_abs[:, 0] = X_prime_abs[:, 0] / np.sum(X_prime_abs[:, 0])
        X_prime_abs[:, 1] = X_prime_abs[:, 1] / np.sum(X_prime_abs[:, 1])

        abskl1 = np.sum(X_prime_abs[:, 0] * np.log((X_prime_abs[:, 0] + 1e-6) / (X_prime_abs[:, 1] + 1e-6)))
        abskl2 = np.sum(X_prime_abs[:, 1] * np.log((X_prime_abs[:, 1] + 1e-6) / (X_prime_abs[:, 0] + 1e-6)))
        abskl = (abskl1 + abskl2)

        return -abskl

    def unmixing(self, coeffs, X):
        # Applies an anmixing matrix build from coeffs to a stereo signal X
        # Returns the resulting stereo signal X_del
        # forward mixing: [[1, a0*z^-delay0],[a1*z^-delay1, 1]]*X
        # Coeffs: total 4: First 2: crosswise atten. coeff., 2nd 2: Crosswise delays
        # inverse mixing matrix: 1/(1-a0*z^-delay0*a1*z^-delay1) * [[1, -a0*z^-delay0],[-a1*z^-delay1, 1]] *X
        # For 2x2 matrix inverse see e.g.: https://www.mathsisfun.com/algebra/matrix-inverse.html
        # IIR Filter 1/(1-a0*z^-delay0*a1*z^-delay1) in lfilter:
        # scipy.signal.lfilter([1], [1, zeros(delay0+delay1-1), -a0*a1])

        X_del = np.zeros(X.shape)
        # Add delays:
        a = np.abs(coeffs[0:2])  # allow only positive values for delay and attenuation, for CG optimizer
        delay = np.abs(coeffs[2:4])
        # print("delay=", delay)

        a0, b0 = self.allp_delayfilt(delay[0])
        a1, b1 = self.allp_delayfilt(delay[1])

        X_del[:, 0] = X[:, 0] - a[0] * scipy.signal.lfilter(b0, a0, X[:, 1])
        X_del[:, 1] = X[:, 1] - a[1] * scipy.signal.lfilter(b1, a1, X[:, 0])

        return X_del

    def allp_delayfilt(self, tau):
        '''
        performs Fractional-delay All-pass Filter
        :param tau: fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
        :type tau: float or int
        :return:
            a: Denumerator of the transfer function
            b: Numerator of the transfer function
            h: Transfer function of the filter
        '''

        # print("tau", tau)
        L = int(tau) + 1
        n = np.arange(0, L)
        # print("n", n)

        a_0 = np.array([1.0])
        a = np.array(np.cumprod(np.divide(np.multiply((L - n), (L - n - tau)), (np.multiply((n + 1), (n + 1 + tau))))))
        a = np.append(a_0, a)  # Denumerator of the transfer function
        # print("Denumerator of the transfer function a:", a)

        b = np.flipud(a)  # Numerator of the transfer function
        # print("Numerator of the transfer function b:", b)

        return a, b


class aires_online_class:


    def __init__(self, Blocksize=512, maxdelay=20, n_blocks_signal_memory=4, state0=None, state1=None, coeffs=[1.0, 1.0, 1.0, 1.0], sigmemory=None, n_iter_pro_block = 2):

        # Initializing the variables
        # The size of block to separate
        self.Blocksize = Blocksize
        # Declare memory for IIR filters
        # maximum expected delay, to fill up coeff array to constant length
        self.maxdelay = maxdelay
        # Number of blocks that will be taken from past together with current block for separation
        # In order to interpolate unmixing filters
        self.n_blocks_signal_memory = n_blocks_signal_memory
        # Initialize the filter memory:
        # fractional delay filter states for the length of the signal memory (n_blocks_signal_memory Blocks)
        self.state0 = np.zeros((self.maxdelay + 1, self.n_blocks_signal_memory + 1))
        self.state1 = np.zeros((self.maxdelay + 1, self.n_blocks_signal_memory + 1))
        # Initial coefficients for separation
        self.coeffs = coeffs
        # Memory for signal separation
        self.sigmemory = np.zeros((self.Blocksize * self.n_blocks_signal_memory, 2))
        # Number iterations per block
        self.n_iter_pro_block = n_iter_pro_block



    def reset_parameters(self):
        print('Default parameters have been Reset')
        # Number of blocks that will be taken from past together with current block for separation
        # In order to interpolate unmixing filters
        self.n_blocks_signal_memory = self.n_blocks_signal_memory
        # Initialize the filter memory:
        # fractional delay filter states for the length of the signal memory (n_blocks_signal_memory Blocks)
        self.state0 = np.zeros((self.maxdelay + 1, self.n_blocks_signal_memory + 1))
        self.state1 = np.zeros((self.maxdelay + 1, self.n_blocks_signal_memory + 1))
        # Memory for signal separation
        self.sigmemory = np.zeros((self.Blocksize * self.n_blocks_signal_memory, 2))


    def aires_online_separation(self, X_mixed_block, rt60):

        self.sigmemory[:-self.Blocksize, :] = self.sigmemory[self.Blocksize:, :]  # shift old samples left
        self.sigmemory[-self.Blocksize:, :] = X_mixed_block  # Write new block on right end

        self.state0[:, :-1] = self.state0[:, 1:]  # shift states 1 left to make space for the newest state
        self.state1[:, :-1] = self.state1[:, 1:]  # shift states 1 left to make space for the newest state

        Xunm_block, self.coeffs, self.state0[:, -1], self.state1[:, -1], process_block_time = \
            self.blockseparationoptimization(self.coeffs,  self.sigmemory, \
                                              self.state0[:, - self.n_blocks_signal_memory - 1], \
                                              self.state1[:, - self.n_blocks_signal_memory - 1], rt60)

        delay = self.coeffs[2:4]
        return Xunm_block, process_block_time


    def minabskl_online(self, X):
        # computes the normalized magnitude of the channels and then applies
        # the Kullback-Leibler divergence

        X_prime = np.copy(X)
        X_prime_abs = np.abs(X_prime)
        # normalize to sum()=1, to make it look like a probability:
        X_prime_abs[:, 0] = X_prime_abs[:, 0] / np.sum(X_prime_abs[:, 0])
        X_prime_abs[:, 1] = X_prime_abs[:, 1] / np.sum(X_prime_abs[:, 1])

        abskl1 = np.sum(X_prime_abs[:, 0] * np.log((X_prime_abs[:, 0] + 1e-6) / (X_prime_abs[:, 1] + 1e-6)))
        abskl2 = np.sum(X_prime_abs[:, 1] * np.log((X_prime_abs[:, 1] + 1e-6) / (X_prime_abs[:, 0] + 1e-6)))
        abskl = (abskl1 + abskl2)

        return -abskl

    def allp_delayfilt(self, tau):
        '''
        performs Fractional-delay All-pass Filter
        :param tau: fractional delay in samples. When 'tau' is a float - sinc function. When 'tau' is an integer - just impulse.
        :type tau: float or int
        :return:
            a: Denumerator of the transfer function
            b: Numerator of the transfer function
            h: Transfer function of the filter
        '''

        import scipy.signal
        # print("tau", type(tau))
        # L = max(1,int(tau)+1) with the +1 the max doesn't make sense anymore
        L = int(tau) + 1
        n = np.arange(0, L)
        # print("n", n)

        a_0 = np.array([1.0])
        a = np.array(np.cumprod(np.divide(np.multiply((L - n), (L - n - tau)), (np.multiply((n + 1), (n + 1 + tau))))))
        a = np.append(a_0, a)  # Denumerator of the transfer function
        # print("Denumerator of the transfer function a:", a)

        b = np.flipud(a)  # Numerator of the transfer function
        # print("Numerator of the transfer function b:", b)

        # Calculate Impulse response of the filter
        '''
        Nh = L * 1  # Length of the transfer function
        impulse_train = np.zeros(Nh + 1)
        impulse_train[0] = 1
        h = scipy.signal.lfilter(b, a, impulse_train)  # impulse response
        # h = h/np.max(h)
        # h = h / np.sqrt(np.sum(h ** 2))
        # print("Len of Demixing Filter", len(h))

        # Plot Impulse response
        # plt.plot(h)
        # plt.show()
        '''

        # print("tau=", tau, "h=", h)
        # print("a,b=", a,b)
        return a, b #, h

    def unmixing_online(self, coeffs, X, state0, state1):
        # Applies an anmixing matrix build from coeffs to a stereo signal X
        # Arguments: coeffs = (attenuation 0, attenuation 1, delay0, delay1)
        # X= Stereo signal from the microphones
        # state0, state1 = states from previous delay filter run, important if applied
        # to consecutive blocks of samples.
        # Returns the resulting (unmixed) stereo signal X_del
        # Unmixing Process:
        # Delayed and attenuated versions of opposing microphones are subtracted:
        # Xdel0= X0- att0 * del0(X1)
        # Xdel1= X1- att1 * del1(X0)

        # print("coeffs =", coeffs)
        X_del = np.zeros(X.shape)
        # maxdelay = maximum expected delay, to fill up coeff array to constant length
        maxdelay = len(state0) - 1
        # Attenuations:
        a = np.clip(coeffs[0:2], -1.5, 1.5)  # limit possible attenuations
        # a = np.abs(coeffs[0:2])
        # delay=np.abs(coeffs[2:4]) #allow only positive values for delay, for CG optimizer
        delay = np.clip(coeffs[2:4], 0, maxdelay)  # allow only range of 0 to maxdelay
        # print("delay=", delay)
        # delay filters for fractional delays:
        # using allpass delay filter:
        a0, b0 = self.allp_delayfilt(delay[0])
        a0 = np.append(a0, np.zeros(maxdelay + 2 - len(a0)))
        b0 = np.append(b0, np.zeros(maxdelay + 2 - len(b0)))
        a1, b1 = self.allp_delayfilt(delay[1])
        a1 = np.append(a1, np.zeros(maxdelay + 2 - len(a1)))
        b1 = np.append(b1, np.zeros(maxdelay + 2 - len(b1)))
        # print("Len of a0, b0:", len(a0), len(b0))
        # Both channels are delayed with the allpass filter:
        y1, state1 = scipy.signal.lfilter(b0, a0, X[:, 1], zi=state1)
        y0, state0 = scipy.signal.lfilter(b1, a1, X[:, 0], zi=state0)
        # Delayed and attenuated versions of opposing microphones are subtracted:
        X_del[:, 0] = X[:, 0] - a[0] * y1
        X_del[:, 1] = X[:, 1] - a[1] * y0
        return X_del, state0, state1

    def blockseparationoptimization(self, coeffs, Xblock, state0, state1, rt60):

        start_block_time = time.time()
        # Reads in a block of a stereo signal, improves the unmixing coefficients,
        # applies them and returns the unmixed block
        # Arguments:
        # coeffs: array of the 4 unmixing coefficients
        # Xblock: Block of the stereo signal to unmix
        # state0, state1: Filter states for the IIR filter, length is maxdelay +1
        # returns:
        # Xunm: the unmixed stereo block resulting from the updated coefficients
        # coeffs: The updated coefficients
        # state0, state1 : the new filter states

        # Simple online optimization, using random directions optimization:
        # Old values 0:
        # coeffs0 = np.copy(coeffs)
        Xunm0, state00, state10 = self.unmixing_online(coeffs, Xblock, state0, state1)  # starting point
        negabskl0 = self.minabskl_online(Xunm0)
        # abskl_buffer.append(negabskl0)
        # print("*************************************************************** 1. negabskl0 =", negabskl0)
        # print("1 coeffs", coeffs)

        # maxdelay = len(state0) - 1

        # print('*************************************', rt60)
        if rt60 == 0.05:
            coeffweights = np.array([0.1, 0.1, 1.0, 1.0]) * 0.6  # RT60 = 0.05
        if rt60 == 0.1:
            coeffweights = np.array([0.05, 0.05, 0.5, 0.5]) * 0.1  # 0.2 #np.array([0.2, 0.2, 0.9, 0.9])     RT60=0.1
        if rt60 == 0.2:
            coeffweights = np.array([0.07, 0.07, 0.7, 0.7]) * 0.4  # np.array([0.2, 0.2, 0.9, 0.9])  RT60=0.2
            # coeffweights = np.array([0.01, 0.01, 0.1, 0.1]) * 4

        for m in range(self.n_iter_pro_block):
            # coeffvariation=(np.random.rand(4)-0.5)*coeffweights
            # np.random.seed(m)
            # print(np.random.rand(4))
            coeffvariation = (np.random.random(4) - 0.5) * coeffweights  # small variation
            # att_max = 3
            # coeffvariation = 0.08*np.array([randrange(-att_max, att_max), randrange(-att_max, att_max), randrange(-self.maxdelay, self.maxdelay), randrange(-self.maxdelay, self.maxdelay)]) * coeffweights  # small variation



            # coeffvariation=np.random.normal(loc=0.0, scale=0.5, size=4) #Gaussian distribution
            # new values 1:
            coeffs_tmp = (coeffs + coeffvariation)  # + coeffs_memory)/2.0

            coeffs_tmp[0:2] = np.clip(coeffs_tmp[0:2], -3.0, 3.0)  # limit possible attenuations
            coeffs_tmp[2:4] = np.clip(coeffs_tmp[2:4], 0, self.maxdelay)  # allow only range of 0 to maxdelay





            Xunm1, state01, state11 = self.unmixing_online(coeffs_tmp, Xblock, state0, state1)
            negabskl1 = self.minabskl_online(Xunm1)

            if negabskl1 < negabskl0:  # New is better
                # print("**************************************************************** NEW  negabskl1, negabskl0=",
                #       negabskl1, negabskl0)
                negabskl0 = negabskl1
                coeffs = coeffs_tmp  # coeffs + coeffvariation
                Xunm = Xunm1
                state0 = state01
                state1 = state11
            else:  # old is better
                # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  OLD IS BETTER")
                # print("2 coeffs", coeffs)
                Xunm = Xunm0
                state0 = state00
                state1 = state10


        process_block_time = time.time() - start_block_time
        return Xunm, coeffs, state0, state1, process_block_time





    def resampling(self, X, samplerate, downsampling_ratio):

        X_4_resampling = np.copy(X)
        # Downsample mixed signal and low-pass it to make cost function more convex, simplify and make convolution faster
        N = downsampling_ratio  # downsampling by N, can be changed
        # low pass filter, 1/N band, before downsamplin:
        lp = scipy.signal.remez(63, bands=[0, 0.32 / N, 0.45 / N, 1], desired=[1, 0], weight=[1, 100], Hz=2)
        X_4_resampling_lp = scipy.signal.lfilter(lp, [1], X_4_resampling, axis=0)
        X_resampled = X_4_resampling_lp[::N, :]
        new_samplerate = int(samplerate / N)

        return X_resampled, new_samplerate


    def create_wiener_filter(self):
        # Open some speech wav file to create Wiener Filter
        samplerate, x = wav.read("fspeech.wav")
        # make x a matrix and transpose it into a column:
        x = np.matrix(x).T
        # additive zero mean white noise (for -2**15<x<+2**15):
        y = x + 0.1 * (np.random.random(np.shape(x)) - 0.5) * 2 ** 15

        # we assume L=10 coefficients for our Wiener filter.
        # 10 to 12 is a good number for speech signals.
        A = np.matrix(np.zeros((100000, 10)))
        for m in range(100000):
            A[m, :] = np.flipud(y[m + np.arange(10)]).T

        # Compute Wiener Filter:
        # Trick: allow filter delay of 5  samples
        # to get better working denoising.
        # This corresponds to the center of our Wiener filter.
        # The desired signal hence is x[5:100005].
        # Observe: Since we have the matrix type, operator
        h = np.linalg.inv(A.T * A) * A.T * x[5:100005]

        # plt.plot(h)
        # plt.xlabel('Sample')
        # plt.ylabel('value')
        # plt.title('Impulse Response of Wiener Filter')
        # plt.show()

        # Save Wiener Filter coeefs to binary file
        pickle.dump(h, open("wiener_filter_coeff.bin", "wb"), 1)

        # return h


    def mutualinformation(self, x, bins):
        # Computes the mutual information I(X;Y) of 2 signals in x
        # Arguments:
        # x: 2d array with 2 columns, 1 column for each signal
        # returns: mutual information absed on the natural logarithm (nats)
        # Computation based on https://en.wikipedia.org/wiki/Mutual_information
        # I(X;Y)= H(X)+H(Y)-H(X,Y)
        # from matplotlib.mlab import entropy

        HX = self.entropy1d(x[:, 0], bins)  # entropy of x

        HY = self.entropy1d(x[:, 1], bins)  # entropy of Y
        # print("HX=", HX, "HY=", HY)
        HXY = self.entropy2d(x, bins)  # joint entropy of X and Y
        return HX + HY - HXY



    def entropy1d(self, x, bins):
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


    def entropy2d(self, x, bins):
        # Function to estimate the joint entropy of 2 signals (in a 2d array)
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

    def playsound(self, audio, samplingRate, channels):
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



def offline_aires_separation(X_mixed):

    # AIRES BSS Configuration

    aires = aires_offline_class()
    aires.maxdelay = 20
    aires.n_iter = 20       # Number iterations in optimization for full signal
    aires.do_downsampling_lowpass = True  # do_lowpass

    start_timer = time.time()
    X_unmixed = aires.aires_offline_separation(X_mixed)  # separate_signals
    process_time = time.time() - start_timer

    return X_unmixed, process_time



def online_aires_separation(X_mixed):

    # AIRES BSS Configuration

    aires = aires_online_class()
    aires.maxdelay = 20
    aires.n_blocks_signal_memory = 4    # Number of previous blocks in memory
    aires.Blocksize = 512               # Size of the processed block
    aires.n_iter_pro_block = 2          # Number iterations per block
    aires.reset_parameters()

    # Number integer blocks in audio signal
    Blocks = max(X_mixed.shape) // aires.Blocksize
    print("Blocks=", Blocks)

    X_unmixed = np.zeros((X_mixed.shape))
    process_block_time_buffer = []

    for m in range(Blocks):
        # Iterate blocks of the signal

        X_mixed_block = X_mixed[m * aires.Blocksize + np.arange(aires.Blocksize), :]

        '''###########################################################'''
        """ Online Separation v.2"""
        # Apply AIRES online separation
        # ToDo: rt60 in the future should be removed
        rt60 = 0.2
        Xunm_block, process_block_time = aires.aires_online_separation(X_mixed_block, rt60)

        process_block_time_buffer.append(process_block_time)
        print(
            "\n ******************************************** \n Time for block calculation:", process_block_time,
            "\n \n")

        print("\n ****************************")
        print("Progress: ", int(100 * m / Blocks), "%")
        print("**************************** \n")
        '''###########################################################'''

        # Store unmixed block:
        X_unmixed[m * aires.Blocksize + np.arange(aires.Blocksize), :] = Xunm_block[-aires.Blocksize:, :]

    return X_unmixed, process_block_time_buffer




if __name__ == '__main__':
    aires = aires_online_class()    # Just for playing back
    used_samplerate = 16000
    samplerate = 16000

    """ Open simulated convolutive mixture """
    f_open_name = "stationary_ss_rt60-0.05_TIMIT_dist-2_mix-0-0.mat"

    x_load = scipy.io.loadmat(f_open_name)
    X_mixed = x_load['mixed_ss']        # Mixed channels
    X_original_rir = x_load['original_rir_ss']  # Original (clean) channels




    # Normilize signal
    X_mixed = X_mixed/np.max(np.abs(X_mixed))

    # Play back mixed channels
    cut_time = 6 * samplerate
    os.system('espeak -s 120 "Play back mixed channels"')
    aires.playsound(X_mixed[:, 0] * 2 ** 15, samplerate, 1)
    aires.playsound(X_mixed[:, 1] * 2 ** 15, samplerate, 1)

    '''###########################################################'''
    # Apply AIRES Blind Source Separation
    # Change here Offline or Online modes

    X_unmixed, process_time = online_aires_separation(X_mixed)
    # X_unmixed, process_time = offline_aires_separation(X_mixed)



    """ ##################################################################### """
    '''Performance Measurement Original VS Unmixed Blockwise'''
    # Like an average SDR measure
    print("Performance Measurement Original VS Unmixed Blockwise in PROGRESS ...")
    start_time = time.time()
    # perf_meas_origVSunmixed = mir_eval_separation.bss_eval_sources_block(X_original_rir.T, X_unmixed.T)
    perf_meas_origVSunmixed = mir_eval_separation.bss_eval_sources(X_original_rir.T, X_unmixed.T)

    print("SDR calculation time", time.time()-start_time)

    # The bigger value of SDR the better
    perf_meas_origVSunmixed = perf_meas_origVSunmixed #[0]
    perm = perf_meas_origVSunmixed[3]
    SDR_full = perf_meas_origVSunmixed[0]  # vector of Signal to Distortion Ratios (SDR)


    print("\n***************************************")
    print("***************************************\n")
    print("AIRES ONLINE FULL to FULL and BLOCK to BLOCK ")
    print("____ Original VS Unmixed ____ \n")
    print("Permutation", perm)
    print("SDR (FULL to FULL):    ", SDR_full)
    # print("SIR:", perf_meas_origVSunmixed[1])
    # sdr_per_block = [np.mean(SDR_block[:, 0]), np.mean(SDR_block[:, 1])]
    # print("Mean SDR (BLOCK to BLOCK)", sdr_per_block)

    # Calculate mean computation time per block of data
    # calc_speed_per_iteration = np.mean(np.asarray(process_block_time_buffer))
    # print("Online AIRES Processing time pro BLOCK:", calc_speed_per_iteration, ", [s]")

    print("Offline AIRES Processing time:", np.sum(process_time), ", [s]")

    print("\n***************************************")
    print("*************************************** \n")



    norm_factor = np.max(np.abs(X_unmixed))
    X_unmixed = X_unmixed * 1.0 / norm_factor

    # Play back separated channels
    os.system('espeak -s 120 "Play back Unmixed channels"')
    aires.playsound(X_unmixed[:, 0] * 2 ** 15, samplerate, 1)
    aires.playsound(X_unmixed[:, 1] * 2 ** 15, samplerate, 1)