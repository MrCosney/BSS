import numpy as np
from scipy import signal
from Player import *
from setups import setups
from Algorithms import mix
from Normalizer import *
from Room import *
import copy
import sys


def main():
    sims, data_sets, algs = setups()
    results = []
    # choose data set
    for sets in data_sets:
        set_name = sets['type']
        set_freq = sets['freq']
        set_data = sets['data']
        X = []
        for wav in set_data:
            X.append(load_wav(wav, set_freq))
        X = normalization(np.array(X))
        # choose mix type
        for sim in sims:
            mix_type = sim['mixture_type']
            sim_name = sim['name']
            # Run algorithms
            for alg in algs:
                print("Running " + alg['name'] + " ...")
                if mix_type == 'linear':
                    Mixd = mix(copy.deepcopy(X[:alg['microphones']]))
                elif mix_type == 'room':
                    Mixd = makeroom(set_freq, copy.deepcopy(X[:alg['microphones']]))
                else:
                    print("Error: Simulation is chosen wrong.")
                    sys.exit()
                if 'options' in alg:
                    opt = alg['options']
                else:
                    opt = None
                alg['Unmix'], alg['state'] = alg['func'](Mixd, alg['state'], opt)
                alg['Unmix'] = normalization(alg['Unmix'])
                alg['Metrics' + sim_name] = evaluate(X, alg['Unmix'])
                #results[alg['name']] = alg
            res = {sim_name: algs}
            results.append(res)


def evaluate(X, S):
    SDR = 1
    SIR = 2
    return {'SDR': SDR, 'SIR': SIR, 'RMSE': rmse(X, S)}


if __name__ == "__main__":
    main()
