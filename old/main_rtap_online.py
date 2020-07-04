from Player import *
from setups import *
from Model import mix
from Normalizer import *
from plots import *
from scipy.io.wavfile import write
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
            X = normalize(np.array(X))

            #fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
            #ax[0, 1].plot(X[0])
            #plt.show()

            # Todo: разобраться с корректной длительностью записи
            data_set['audio_duration'] = round(len(X[0]) / data_set['fs'], 1)

            # 3. Perform environment simulation (mix signals)
            filtered, mixed, sim = mix(X, sim, data_set)

            # 3.1 Rework online-recording shape for convolutive method
            if sim['mix_type'] == 'convolutive':
                mixed = rework_conv(mixed, sim)

            # TODO: не успел сохранять графики, батарейки сдохли на колонках
            plot_filtered(filtered)
            # 4.1 Normalize Recorded Audio
            for i in range(len(mixed)):
                mixed[i] = normalize(mixed[i])
            sim['mixed'] = mixed
            # 4.2 Normalize Filtered data
            # sim['filtered'] = normalization(filtered)
            # TODO: ложус спат не успел посмотреть, после этого пункт ниже, все филтеред срезы становятся одинаковые
            for f in filtered:
                filtered[...] = normalize(f)
            sim['filtered'] = filtered

            # 5 Create folders for save data
            dir_name = "".join(("Audio/Unmixed/", sim['name']))
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

            # 6. Run algorithms
            print('\n\033[35mSeparation process:\033[0m')
            for alg in sim['algs']:
                # TODO: Fix correct work with chunk_size == STFT
                print("\tSeparation by " '\033[33m' + alg['name'], '\033[0m' + "in " + sim['name'] +
                      " with " + str(sim['chunk_size']) + " Chunk size" "....")

                temp_data = []
                for i in range(len(mixed)):
                    unmixed, alg['state'] = alg['func'](mixed[i], alg['state'], alg.get('options'))
                    temp_data.append(unmixed)

                # 6.1 Combine all reconstructed chunks into data and save into .wav files
                recovered_data = np.concatenate(temp_data, axis=1)

                alg_dir = "".join((dir_name, "/", alg['name'], "/"))
                if not os.path.isdir(alg_dir):
                    os.mkdir(alg_dir)
                for i in range(recovered_data.shape[0]):
                    write("".join((alg_dir, data_set['file_names'][i])), data_set['fs'], np.float32(recovered_data[i]))

                # 6.2 Normalize the unmixed data and calculate metrics
                alg['unmixed'] = normalize(recovered_data)
                alg['metrics'] = {data_set['type']: evaluate(X, sim['filtered'], alg['unmixed'])}

            # delete temporary "mixed" array form dict
            del sim['mixed']

    # Collect all metrics into new dictionary, display in in console with correct view and plot the results in folder
    dict_data = rework_dict(sims)
    plot_metrics(dict_data)


def evaluate(original: np.ndarray, filtered: np.ndarray, unmixed: np.ndarray) -> dict:
    """Evaluate the metrics"""
    ref = np.moveaxis(filtered, 1, 2)
    Ns = np.minimum(unmixed.shape[1], ref.shape[1])
    Sn = np.minimum(unmixed.shape[0], ref.shape[0])
    SDR, SIR, SAR, P = bss_eval_sources(ref[: Sn, :Ns, 0], unmixed[: Sn, :Ns])
    # TODO: RMSE was removed because of Singular Matrix error, uncomment for check
    #return {'SDR': SDR, 'SIR': SIR, 'SAR': SAR, 'P': P, 'RMSE': rmse(original, unmixed)}
    return {'SDR': SDR, 'SIR': SIR, 'SAR': SAR, 'P': P}


if __name__ == "__main__":
    main()
