import numpy as np
from sklearn.metrics import mean_squared_error
import copy

def normalization(s):
    if s.shape[0] < 7:
        for i in range(s.shape[0]):
            s[i] = ((np.float64(s[i]) - np.float64(min(s[i]))) * 2) / (np.float64(np.max(s[i])) - np.float64(min(s[i]))) - 1
    else:
        s = ((np.float64(s) - np.float64(min(s))) * 2) / (np.float64(np.max(s)) - np.float64(min(s))) - 1
    return s

def rmse(original, unmixed):
    '''Return the array with RMSE for each source signal'''
    #Compute max length of the vector and resize others with it for coreect RMSE calculation
    sources = original.shape[0]
    mics = unmixed.shape[0]
    length = 0
    for i in range(mics):
        if len(original[i]) > length:
            length = len(original[i])
        if len(unmixed[i]) > length:
            length = len(unmixed[i])

    #pad zeroes to make both equal
    orig_t = np.zeros((sources, length))
    unmixed_t = np.zeros((mics, length))
    for i in range(mics):
        orig_t[i][:original[i].shape[0]] = original[i]
        unmixed_t[i][:unmixed[i].shape[0]] = unmixed[i]

    #find all possible options for RMSE
    rmse_t = []
    for i in range(mics):
        for k in range(mics):
            rmse_t.append(np.sqrt(mean_squared_error(orig_t[i], unmixed_t[k])))
        for z in range(mics):
            rmse_t.append(np.sqrt(mean_squared_error(orig_t[i], unmixed_t[z] * -1)))
    # calculate RMSE of algorithm
    rms = []
    for i in range(mics):
        rms.append(rmse_t.pop(rmse_t.index(min(rmse_t))))
    rmse = np.mean(rms)
    return np.round(rmse, 4)
