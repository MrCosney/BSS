
from Player import *
from setups import setups
from Model import mix
from Normalizer import *
from plots import *
from mir_eval.separation import bss_eval_sources


def main():
    sims, data_sets = setups()

    # Iterate over simulations
    for sim in sims:
        M = sim['microphones']

        # Choose data set
        for data_set in sim['data_sets']:

            # 1. Load source signals
            X = []
            for wav in data_set['data']:
                if len(X) == M:
                    break
                if type(wav) == str:
                    X.append(load_wav(wav, data_set['fs']))
                else:
                    X.append(wav)

            # 2. Normalize & format source signals
            X = form_source_matrix(X)
            X = normalization(np.array(X))

            # 3. Perform environment simulation (mix signals)
            sim['options']['fs'] = data_set['fs']  # sampling frequency depends on the used data set
            filtered, mixed, sim['mix_additional_outputs'] = mix(X, sim['mix_type'], sim['options'])

            # 4. Normalize filtered & mixed arrays
            sim['mixed'] = normalization(mixed)
            for f in filtered:
                filtered[...] = normalization(f)
            sim['filtered'] = filtered

            # 5. Run algorithms
            for alg in sim['algs']:

                # ToDo: this check probably needs to be somewhere else
                # ToDo: this for skip Gen.Signals for ILRMA alg
                if alg['name'].find('ILRMA') == 0 and data_set['type'] == 'Gen Signals':
                    print('Artificially generated signals are not used with ILRMA')
                    continue

                print("Running " + alg['name'] + " in " + sim['name'] + "....")
                unmixed, alg['state'] = alg['func'](sim['mixed'], alg['state'], alg.get('options'))
                alg['unmixed'] = normalization(unmixed)

                # if alg['name'] == 'AIRES' and sim['name'] == 'Convolutive_2_7':
                #     play(alg['unmixed'][0] * 15000)
                #     play(alg['unmixed'][1] * 15000)

                alg['metrics'] = {data_set['type']: evaluate(X, sim['filtered'], alg['unmixed'])}

            # delete temporary "mixed" array form dict
            del sim['mixed']

    # Collect all metrics into new dictionary, display in in console with correct view and plot the results in folder
    dict_data = rework_dict(sims)
    plot_metrics(dict_data)


def evaluate(original: np.ndarray, filtered: np.ndarray, unmixed: np.ndarray) -> dict:
    ref = np.moveaxis(filtered, 1, 2)
    Ns = np.minimum(unmixed.shape[1], ref.shape[1])
    SDR, SIR, SAR, P = bss_eval_sources(ref[:, :Ns, 0], unmixed[:, :Ns])
    return {'SDR': SDR, 'SIR': SIR, 'SAR': SAR, 'P': P, 'RMSE': rmse(original, unmixed)}


if __name__ == "__main__":
    main()
