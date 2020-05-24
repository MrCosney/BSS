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
    # choose sim type
    for sim in sims:
        mix_type = sim['mixture_type']
        sim_name = sim['name']
        # choose data set
        for sets in data_sets:
            X = []
            for wav in sets['data']:
                X.append(load_wav(wav, sets['freq']))
            X = normalization(np.array(X))
            # Run algorithms
            for alg in algs:
                print("Running " + alg['name'] + " ...")
                # choose mix type
                if mix_type == 'linear':
                    alg['Mix'] = mix(copy.deepcopy(X[:alg['microphones']]))
                elif mix_type == 'room':
                    alg['Mix'] = makeroom(sets['freq'], copy.deepcopy(X[:alg['microphones']]))
                else:
                    print("Error: Simulation is chosen wrong.")
                    sys.exit()
                alg['Mix'] = normalization(alg['Mix'])
                if 'options' in alg:
                    opt = alg['options']
                else:
                    opt = None
                alg['Unmix'], alg['state'] = alg['func'](alg['Mix'], alg['state'], opt)
                alg['Unmix'] = normalization(alg['Unmix'])
                alg['Metrics' + sim_name] = evaluate(X, alg['Unmix'])
                #results[alg['name']] = alg
            res = {sim_name + sets['type']: algs}
            results.append(copy.deepcopy(res))


def evaluate(X, S):
    SDR = 1
    SIR = 2
    return {'SDR': SDR, 'SIR': SIR, 'RMSE': rmse(X, S)}


if __name__ == "__main__":
    main()
