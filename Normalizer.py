import numpy as np
from sklearn.metrics import mean_squared_error

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
    temp = np.copy(original)
    temp_u = np.copy(unmixed)
    length = 0
    for i in range(unmixed.shape[0]):
        if len(temp[i]) > length:
            length = len(temp[i])
        if len(unmixed[i]) > length:
            length = len(unmixed[i])

    for i in range(unmixed.shape[0]):
        temp[i] = np.resize(temp[i], (length, ))
        unmixed[i] = np.resize(temp_u[i], (length, ))

    #find all possible options for RMSE
    rmse_t = []
    for i in range(unmixed.shape[0]):
        for k in range(unmixed.shape[0]):
            rmse_t.append(np.sqrt(mean_squared_error(temp[i], unmixed[k])))
        for z in range(unmixed.shape[0]):
            rmse_t.append(np.sqrt(mean_squared_error(temp[i], unmixed[z] * -1)))
    # calculate RMSE of algorithm
    rms = []
    for i in range(unmixed.shape[0]):
        rms.append(rmse_t.pop(rmse_t.index(min(rmse_t))))
    rmse = np.mean(rms)
    return np.round(rmse, 4)
