import numpy as np
from sklearn.metrics import mean_squared_error
import time


def normalization(s):
    n_s = ((np.float64(s) - np.float64(min(s))) * 2) / (np.float64(np.max(s)) - np.float64(min(s))) - 1
    return n_s

def normalization2(s):
    n_s = ((np.float64(s) - np.float64(min(s))) * 2) / (np.float64(np.max(s)) - np.float64(min(s))) - 1
    return n_s

def rmse(s1, s2, s3, un):
    '''Return the array with RMSE for each source signal'''
    length = max(len(s1), len(s2), len(s3), len(un[0]))
    s1.resize((length, 1), refcheck=False)
    s2.resize((length, 1), refcheck=False)
    s3.resize((length, 1), refcheck=False)

    n_d1 = ((np.float64(s1) - np.float64(min(s1))) * 2) / (np.float64(np.max(s1)) - np.float64(min(s1))) - 1
    n_d2 = ((np.float64(s2) - np.float64(min(s2))) * 2) / (np.float64(np.max(s2)) - np.float64(min(s2))) - 1
    n_d3 = ((np.float64(s3) - np.float64(min(s3))) * 2) / (np.float64(np.max(s3)) - np.float64(min(s3))) - 1

    n_un1 = ((np.float64(un[0]) - np.float64(min(un[1]))) * 2) / (np.float64(np.max(un[0])) - np.float64(min(un[0]))) - 1
    n_un2 = ((np.float64(un[1]) - np.float64(min(un[0]))) * 2) / (np.float64(np.max(un[1])) - np.float64(min(un[1]))) - 1
    n_un3 = ((np.float64(un[2]) - np.float64(min(un[2]))) * 2) / (np.float64(np.max(un[2])) - np.float64(min(un[2]))) - 1

    rmse_1 = np.sqrt(mean_squared_error(n_d1, n_un1))
    rmse_2 = np.sqrt(mean_squared_error(n_d2, n_un2))
    rmse_3 = np.sqrt(mean_squared_error(n_d3, n_un3))
    rmse = np.array([rmse_1, rmse_2, rmse_3])
    return rmse

def rmse_s(s1, s2, s3, unmix):
    rmse_1 = np.sqrt(mean_squared_error(s1, unmix[0]))
    rmse_2 = np.sqrt(mean_squared_error(s2, unmix[1]))
    rmse_3 = np.sqrt(mean_squared_error(s3, unmix[2]))
    rmse = np.array([rmse_1, rmse_2, rmse_3])
    return rmse

def swap_lines(object,fr, on):
    '''Swap 0 and 1 lines . For some algorithms'''
    temp = np.copy(object)
    object[fr] = temp[on]
    object[on] = temp[fr]
    return object

def rmse2(s1, s2, un):
    '''Return the array with RMSE for each source signal'''
    length = max(len(s1), len(s2), len(un[0]))
    s1.resize((length, 1), refcheck=False)
    s2.resize((length, 1), refcheck=False)
    un[0].resize((length, 1), refcheck=False)
    un[1].resize((length, 1), refcheck=False)

    n_d1 = ((np.float64(s1) - np.float64(min(s1))) * 2) / (np.float64(np.max(s1)) - np.float64(min(s1))) - 1
    n_d2 = ((np.float64(s2) - np.float64(min(s2))) * 2) / (np.float64(np.max(s2)) - np.float64(min(s2))) - 1

    n_un1 = ((np.float64(un[0]) - np.float64(min(un[1]))) * 2) / (np.float64(np.max(un[0])) - np.float64(min(un[0]))) - 1
    n_un2 = ((np.float64(un[1]) - np.float64(min(un[0]))) * 2) / (np.float64(np.max(un[1])) - np.float64(min(un[1]))) - 1

    rmse_1 = np.sqrt(mean_squared_error(n_d1, n_un1))
    rmse_2 = np.sqrt(mean_squared_error(n_d2, n_un2))
    rmse = np.array([rmse_1, rmse_2])
    return rmse