import numpy as np
from sklearn.metrics import mean_squared_error

def normalization(s):
    if s.shape[0] < 7:
        for i in range(s.shape[0]):
            s[i] = ((np.float64(s[i]) - np.float64(min(s[i]))) * 2) / (np.float64(np.max(s[i])) - np.float64(min(s[i]))) - 1
    else:
        s = ((np.float64(s) - np.float64(min(s))) * 2) / (np.float64(np.max(s)) - np.float64(min(s))) - 1
    return s

def rmse(original, unmixed, name):
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
    rmse = []
    for i in range(unmixed.shape[0]):
        rmse.append(np.sqrt(mean_squared_error(temp[i], unmixed[i])))
    rmse = np.array(rmse)

    file_1 = open(name, "w")
    file_1.write(str(np.mean(rmse)))
    file_1.close()
    return rmse
