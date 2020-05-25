import numpy as np
from scipy import signal
from Player import *
from setups import setups
from Algorithms import mix
from Normalizer import *
from MakeRoom import *
import copy
import sys


def main():
    sims, data_sets = setups()
    # choose sim type
    for sim in sims:
        mix_type = sim['mixture_type']
        sim_name = sim['name']
        mics = sim['microphones']
        # choose data set
        for sets in sim['data_sets']:
            X = []
            for wav in sets['data']:
                if len(X) == mics:
                    break
                if type(wav) == str:
                    X.append(load_wav(wav, sets['freq']))
                else:
                    X.append(wav)
            X = normalization(np.array(X))
            # choose mix type
            mix_data = sim['mix_data']
            if mix_type == 'linear':
                sim['Mix_data'] = mix(copy.deepcopy(X))             #can be removed?
                mix_data.update({sets['type']: mix(copy.deepcopy(X))})
            elif mix_type == 'room':
                a =1
                #alg['Mix_data'] = makeroom(sets['freq'], copy.deepcopy(X), alg['options'])
            else:
                print("Error: Simulation is chosen wrong.")
                sys.exit()
            sim['Mix_data'] = normalization(sim['Mix_data'])
            # Run algorithms
            for alg in sim['algs']:
                metrics = alg['Metrics']
                print("Running " + alg['name'] + " ...")
                # choose mix type
                if 'options' in alg:
                    opt = alg['options']
                else:
                    opt = None
                alg['Unmix'], alg['state'] = alg['func'](sim['Mix_data'], alg['state'], opt)
                alg['Unmix'] = normalization(alg['Unmix'])
                metrics.update({sets['type']: evaluate(X, alg['Unmix'])})
            #delete temp Mix_data form dict
            del sim['Mix_data']


def evaluate(X, S):
    SDR = 1
    SIR = 2
    return {'SDR': SDR, 'SIR': SIR, 'RMSE': rmse(X, S)}


if __name__ == "__main__":
    main()
