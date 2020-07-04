from Player import *
from setups import *
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
            filtered, mixed, sim = mix(X, sim, data_set)

            # 4. Normalize filtered & mixed arrays
            sim['mixed'] = normalization(mixed)
            for f in filtered:
                filtered[...] = normalization(f)
            sim['filtered'] = filtered

            # 5 Padding and splitting into chunks
            mixed = sim['mixed']
            n_chunks = (len(mixed[0]) // sim['chunk_size']) * sim['chunk_size']
            pad_number = sim['chunk_size'] - (len(mixed[0]) - n_chunks)
            temp_mix = np.zeros((mixed.shape[0], len(mixed[1]) + np.round(pad_number)), dtype=float)
            temp_mix[:mixed.shape[0], :mixed.shape[1]] = mixed

            mix_queue = []
            # fill list of chunks with fragmented mix data
            for chunk in range(int(len(temp_mix[0]) / sim['chunk_size'])):
                mix_queue.append(temp_mix[:mixed.shape[0], sim['chunk_size'] * chunk: sim['chunk_size'] * (chunk + 1)])

            # 5. Run algorithms
            print('\033[35mSeparation process:\033[0m')
            for alg in sim['algs']:
                if alg['name'].find('ILRMA') == 0 and data_set['type'] == 'Gen Signals':
                    print('Artificially generated signals are not used with ILRMA')
                    continue
                print("\tSeparation by" + alg['name'] + " in " + sim['name'] + " with " + str(sim['chunk_size']) + " Chunk size" "....")
                temp_data = []
                for i in range(len(mix_queue)):
                    unmixed, alg['state'] = alg['func'](mix_queue[i], alg['state'], alg.get('options'))
                    temp_data.append(unmixed)
                #combine all reconstructed chunks into data
                recovered_data = np.concatenate(temp_data, axis=1)
                #play(recovered_data[0] * 10000)
                alg['unmixed'] = normalization(recovered_data)
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