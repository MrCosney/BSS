from AlgsTest import *
import numpy as np


def main():
    sims = [
        {
            'name': 'Linear',
            'mixture_type': 'linear'
        },
        {
            'name': 'Convolutive',
            'mixture_type': 'room'
        }
    ]

    algs = [
        {
            'name': 'JADE 1 iter',
            'func': some_alg1,
            'options': {'Niter': 1, 'opt2': 2},
            'state': {}
        },
        {
            'name': 'JADE 10 iter',
            'func': some_alg2,
            'options': {'d': [3, 5]},
            'state': {}
        }
    ]

    for sim in sims:
        mix_type = sim['mixture_type']
        sim_name = sim['name']
        # Load wavs
        # ... -> S

        S = np.zeros([7, 2048])
        X = np.zeros([7, 2048])

        if (mix_type == 'linear'):
            # Create matrix and mix
            pass
        else:
            # Create room and mix
            pass

        # Create scenario
        # ... -> ...
        # Mix sources
        # ... -> X

        # Run algorithms
        for alg in algs:
            print("Running " + alg['name'])
            f = alg['func']
            state = alg['state']
            alg['Y'], alg['state'] = f(X, state, alg['options'])
            alg['metrics_' + sim_name] = evaluate(S, alg['Y'])

    a = 1


def evaluate(S, Y):
    SDR = 1
    SIR = 2
    RMSE = 3
    return {'SDR': SDR, 'SIR': SIR, 'RMSE': RMSE}

if __name__ == "__main__":
    main()
