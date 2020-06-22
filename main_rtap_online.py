from Player import *
from setups import *
from Model import mix
from Normalizer import *
from plots import *
from mir_eval.separation import bss_eval_sources
from RecorderClass import Recorder
import threading


def main():
    sims, data_sets = setups()

    # Iterate over simulations
    for sim in sims:
        M = sim['microphones']

        for data_set in sim['data_sets']:
            X = []
            for wav in data_set['data']:
                if len(X) == M:
                    break
                if type(wav) == str:
                    X.append(load_wav(wav, data_set['fs']))
                else:
                    X.append(wav)

            X = form_source_matrix(X)

            #calculate the max duration of audio for recording #Todo:fix for more fit
            data_set['audio_duration'] = len(X[0]) / data_set['fs']

            #Make the threads for Recorder and Player
            #TODO: Maybe rewrite with Multiprocessing Pool class, I tried but it wait the thread for some reason (Commented version on the bottom of this page )
            idx = speakers_device_idx()
            recorder = Recorder(kwargs=({'fs': data_set['fs'],
                                         'chunk_size': sim['chunk_size'],
                                         'audio_duration': data_set['audio_duration']}))
            rec = threading.Thread(target=recorder._record)
            s1 = threading.Thread(target=play, args=(X[0], idx[0]))
            #s2 = threading.Thread(target=play, args=(X[1], idx[1]))

            #start threads and wait till last speaker is done
            rec.start()
            s1.start()
            #s2.start()
            rec.join()

            #Collect recorded data from the Recorder in chunks representation
            rec_data = recorder._data

            # 4. Normalize filtered & mixed arrays
            #TODO: Stopped here
            a/5
            # 5. Run algorithms
            for alg in sim['algs']:
                if alg['name'].find('ILRMA') == 0 and data_set['type'] == 'Gen Signals':
                    print('Artificially generated signals are not used with ILRMA')
                    continue
                print("Running " + alg['name'] + " in " + sim['name'] + " with " + str(sim['chunk_size']) + " Chunk size" "....")
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


# recorder = Recorder1(kwargs=(data_set['fs'], sim['chunk_size'], data_set['audio_duration']))
# pool = Pool(
#     processes=X.shape[0] + 2)  # TODO: Make .shape[0] processes + 1 for rec + 1 for main(P.s not sure have to check
# rec_data = pool.apply_async(recorder._record)
# z = rec_data.get(timeout=5)