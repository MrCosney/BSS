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

        # 1. Load source signals
        for data_set in sim['data_sets']:
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

            # 4. Produce audio data from speakers and record it by the board (Real Simulation) #Todo:fix for more fit
            data_set['audio_duration'] = len(X[0]) / data_set['fs']
            rec_data = play_and_record(X, data_set, sim)

            # 5. Normalize filtered & mixed arrays
                # 5.1 Normalize Recorded Audio
            for i in range(len(rec_data)):
                rec_data[i] = normalization(rec_data[i])
            sim['real_mixed'] = rec_data

                # 5.2 Normalize Simulated Audio and filtered
            sim['mixed'] = normalization(mixed)
            for f in filtered:
                filtered[...] = normalization(f)
            sim['filtered'] = filtered

            # 6. Run algorithms
            print('\n\033[35mSeparation process:\033[0m')
            for alg in sim['algs']:
                if alg['name'].find('ILRMA') == 0 and data_set['type'] == 'Gen Signals':
                    print('Artificially generated signals are not used with ILRMA')
                    continue
                print("\tSeparation by " '\033[33m' + alg['name'], '\033[0m' + "in " + sim['name'] + " with " + str(sim['chunk_size']) + " Chunk size" "....")
                temp_data = []
                #TODO: Добавить внешний цикл для анмиксинга и rec_data и mixed
                for i in range(len(rec_data)):
                    unmixed, alg['state'] = alg['func'](rec_data[i], alg['state'], alg.get('options'))
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


# recorder = Recorder1(kwargs=(data_set['fs'], sim['chunk_size'], data_set['audio_duration']))
# pool = Pool(
#     processes=X.shape[0] + 2)  # TODO: Make .shape[0] processes + 1 for rec + 1 for main(P.s not sure have to check
# rec_data = pool.apply_async(recorder._record)
# z = rec_data.get(timeout=5)