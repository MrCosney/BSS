import numpy as np
from scipy import signal
from Player import *
from setups import setups
from algs.Algorithms import mix
from Normalizer import *
from MakeRoom import *
import copy
import sys
from pprint import pprint


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
                sim['Mix_data'], sim['Room_shape'] = makeroom(sets['freq'], copy.deepcopy(X), sim['options'])
                mix_data.update({sets['type']: mix(copy.deepcopy(X))})
            else:
                print("Error: Simulation is chosen wrong.")
                sys.exit()
            sim['Mix_data'] = normalization(sim['Mix_data'])
            # Run algorithms
            for alg in sim['algs']:

                if alg['name'].find('ILRMA') == 0 and sets['type'] == 'Gen Signals':
                    continue
                metrics = alg['Metrics']
                print("Running " + alg['name'] + " in " + sim_name + "....")
                # choose mix type
                if 'options' in alg:
                    opt = alg['options']
                else:
                    opt = None
                alg['Unmix'], alg['state'] = alg['func'](sim['Mix_data'], alg['state'], opt)
                alg['Unmix'] = normalization(alg['Unmix'])
                if alg['name'] == 'AIRES' and sim['name'] == 'Convolutive_2_7':
                    play(alg['Unmix'][0] * 15000)
                    play(alg['Unmix'][1] * 15000)
                metrics.update({sets['type']: evaluate(copy.deepcopy(X), copy.deepcopy(alg['Unmix']))})
            #delete temp Mix_data form dict
            del sim['Mix_data']

    #Collect all metrics into new dictionary and display in in console with correct view =)
    results = {}
    for sim in sims:
        dic = {sim['name']: {}}
        dic2 = dic[sim['name']]
        for alg in sim['algs']:
            dic2.update({alg['name']: alg['Metrics']})
        results.update(dic)
    print("______________________________")
    for key, value in results.items():
        print("{0} :".format(key))
        for alg, data in value.items():
            print("\t{0} :".format(alg))
            for audio_type, metrics in data.items():
                print("\t\t{0}: {1}".format(audio_type, metrics))
    print("______________________________")

def evaluate(X, S):
    SDR = 1
    SIR = 2
    return {'SDR': SDR, 'SIR': SIR, 'RMSE': rmse(X, S)}


if __name__ == "__main__":
    main()
