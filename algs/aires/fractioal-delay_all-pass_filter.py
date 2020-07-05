# -*- coding: utf-8 -*-
#!/usr/bin/env python2

"""
Created on September 19, 2018

@author: Oleg Golokolenko

1. This code performs Fractional-delay All-pass Filter
Based on tutorial of Ivan W. Selesnick May 20, 2010

"""



import numpy as np
import scipy
import scipy.signal as sp
import matplotlib.pyplot as plt




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


    L = int(tau)+1
    print("L", L)
    n = np.arange(0,L)
    # print("n", n)


    a_0 = np.array([1.0])
    a = np.array(np.cumprod( np.divide(np.multiply((L - n), (L - n - tau)) , (np.multiply((n + 1), (n + 1 + tau))) ) ))
    a = np.append(a_0, a)   # Denumerator of the transfer function
    # print("Denumerator of the transfer function a:", a)

    b = np.flipud(a)     # Numerator of the transfer function
    # print("Numerator of the transfer function b:", b)


    # Calculate Impulse response of the filter
    Nh = L*5 # Length of the transfer function
    impulse_train = np.zeros(Nh+1)
    impulse_train[0] = 1
    h = scipy.signal.lfilter(b, a, impulse_train)   # Transfer function

    return a, b, h



if __name__ == "__main__":

    fontsize = 10


    # Create Delay-Filter
    tau0 = 3.5
    print("tau", tau0)
    a0, b0, h0 = frac_delay_filt(tau0)
    print(a0)
    print(b0)


    # Plot Delay-Filter Impulse Response
    plt.stem(h0)
    # plt.plot(h)
    plt.title('Fractional-delay All-pass Filter Impulse Response', fontsize=fontsize)
    plt.show()


    # Create Test Signal
    N = 100
    x = np.multiply(np.blackman(N), np.cos(0.2*np.pi*np.arange(0,N)))


    # Filtering of the Test Signal
    x_prime0 = scipy.signal.lfilter(b0, a0, x)



    # Plots
    plt.plot(x, label='Original Signal')
    plt.plot(x_prime0, label='Delayed Signal 0')
    plt.legend(bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.)

    plt.title('Fractional-delay All-pass Filter Test ', fontsize=fontsize)
    plt.ylabel('Signal amplitude', fontsize=fontsize)
    plt.xlabel('Time', fontsize=fontsize)

    plt.show()



