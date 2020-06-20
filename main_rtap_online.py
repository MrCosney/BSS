from Player import *
from setups import *
from Model import mix
from Recorder import *
from Normalizer import *
from plots import *
from mir_eval.separation import bss_eval_sources
import threading
from multiprocessing import Queue


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

            idx = speakers_device_idx()
            rec = threading.Thread(target=Recorder)
            s1 = threading.Thread(target=play, args=(X[0], idx[0]))
            s2 = threading.Thread(target=play, args=(X[1], idx[1]))
            #start threads and wait till last speaker is done
            rec.start()
            s1.start()
            s2.start()
            s2.join()

            # 4. Normalize filtered & mixed arrays
            play(a * 10000 , device_idx_list[0])

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