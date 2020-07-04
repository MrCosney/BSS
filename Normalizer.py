import numpy as np
from sklearn.metrics import mean_squared_error
import copy


def form_source_matrix(S_input: list) -> np.ndarray:
    S = copy.deepcopy(S_input)
    l_max = max(len(s) for s in S)

    for s in S:
        s.resize((1, l_max), refcheck=False)

    return np.vstack(S)


def normalize_rowwise(S: np.ndarray) -> np.ndarray:
    S = np.float64(S)
    for i in range(S.shape[0]):
        S[i] = normalize(S[i])
    return S


def normalize(s: np.ndarray) -> np.ndarray:
    s = np.float64(s)
    span = np.max(s) - np.min(s)
    span = 1 if span == 0 else span  # safety check to avoid division by zero
    minim = np.min(s)
    s = ((s - minim) * 2) / span - 1
    return s


def normalize_old(S: np.ndarray) -> np.ndarray:
    if S.shape[0] < 7:
        for i in range(S.shape[0]):
            S[i] = ((np.float64(S[i]) - np.float64(min(S[i]))) * 2) / (np.float64(np.max(S[i])) - np.float64(min(S[i]))) - 1
    else:
        S = ((np.float64(S) - np.float64(min(S))) * 2) / (np.float64(np.max(S)) - np.float64(min(S))) - 1
    return S


def rmse(original, unmixed):
    """Return the array with RMSE for each source signal"""
    # Compute max length of the vector and resize others with it for correct RMSE calculation
    sources = original.shape[0]
    mics = unmixed.shape[0]
    length = 0
    for i in range(mics):
        if len(original[i]) > length:
            length = len(original[i])
        if len(unmixed[i]) > length:
            length = len(unmixed[i])

    # pad zeroes to make both equal
    orig_t = np.zeros((sources, length))
    unmixed_t = np.zeros((mics, length))
    for i in range(mics):
        orig_t[i][:original[i].shape[0]] = original[i]
        unmixed_t[i][:unmixed[i].shape[0]] = unmixed[i]

    # find all possible options for RMSE
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


def rework_conv(mixed: np.ndarray, sim: dict) -> list:
    """Rework the convolutive data into chunks, for Real-Time emulation"""

    # Parameters
    S_CH = sim['chunk_size']
    M = mixed.shape[0]
    Ns = mixed.shape[1]

    # Padding
    Ns_CH = (Ns // S_CH) * S_CH
    pad_number = S_CH - (Ns - Ns_CH)
    mixed_padded = np.zeros((M, Ns + np.round(pad_number)), dtype=float)
    mixed_padded[:M, :Ns] = mixed

    # Splitting into chunks - fill list of chunks with fragmented mixed data
    mixed_queue = []
    for chunk in range(int(len(mixed_padded[0]) / S_CH)):
        mixed_queue.append(mixed_padded[:M, S_CH * chunk: S_CH * (chunk + 1)])

    return mixed_queue
