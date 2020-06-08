import numpy as np
from sklearn.metrics import mean_squared_error
import copy


def form_source_matrix(S_input: list) -> np.ndarray:
    S = copy.deepcopy(S_input)
    l_max = max(len(s) for s in S)

    for s in S:
        s.resize((1, l_max), refcheck=False)

    return np.vstack(S)


def normalization(S: np.ndarray) -> np.ndarray:
    if S.shape[0] < 7:
        for i in range(S.shape[0]):
            S[i] = ((np.float64(S[i]) - np.float64(min(S[i]))) * 2) / (np.float64(np.max(S[i])) - np.float64(min(S[i]))) - 1
    else:
        S = ((np.float64(S) - np.float64(min(S))) * 2) / (np.float64(np.max(S)) - np.float64(min(S))) - 1
    return S


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
