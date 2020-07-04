from Player import *
from setups import *
from Model import mix
from Normalizer import *
from plots import *
from scipy.io.wavfile import write
from mir_eval.separation import bss_eval_sources
from datetime import datetime
from typing import Tuple


def main():
    sims, data_sets = setups()

    dir_sims, dir_plots = create_folders()

    # Iterate over simulations
    for sim in sims:
        # Create folders to save data
        dir_sim, dir_sim_unmixed, dir_sim_plots = sim_create_folders(sim, dir_sims)

        # 1. Load source signals
        for data_set in sim['data_sets']:
            S = []
            for wav in data_set['data']:
                if type(wav) == str:
                    S.append(load_wav(wav, data_set['fs']))
                else:
                    S.append(wav)

            # 2. Normalize & format source signals
            S = form_source_matrix(S)
            S = normalize_rowwise(np.array(S))

            # fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
            # ax[0, 1].plot(X[0])
            # plt.show()

            # Todo: разобраться с корректной длительностью записи
            data_set['audio_duration'] = round(S.shape[1] / data_set['fs'], 1)

            # 3. Perform environment simulation (mix signals)
            filtered, mixed, sim = mix(S, sim, data_set)

            # 4. Normalize filtered & mixed arrays
            mixed = normalize(mixed)
            for f in filtered:
                filtered[...] = normalize(f)
            sim['filtered'] = filtered
            sim['mixed'] = mixed

            # TODO: не успел сохранять графики, батарейки сдохли на колонках
            plot(filtered)

            # 4.1. Create list of chunks (online version only)
            if sim['run_type'] == 'online':
                mixed_queue = rework_conv(mixed, sim)
            else:
                mixed_queue = []

            # 5. Run algorithms
            print('\033[35mSeparation process:\033[0m')
            for alg in sim['algs']:
                # TODO: Fix correct work with chunk_size == STFT

                if alg['name'].find('ILRMA') == 0 and data_set['type'] == 'Gen Signals':
                    print('Warning: artificially generated signals are not used with ILRMA, skipping...')
                    continue

                print("\tSeparation by {} in simulation {} with chunk_size={} ..."
                      .format(alg['name'], sim['name'], sim['chunk_size']))

                # 5.1 Run given algorithm (online or batch)
                if sim['run_type'] == 'online':
                    unmixed = []
                    for chunk in mixed_queue:
                        unmixed_chunk, alg['state'] = alg['func'](chunk, alg['state'], alg.get('options'))
                        unmixed.append(unmixed_chunk)
                    # combine all reconstructed chunks
                    unmixed = np.concatenate(unmixed, axis=1)
                elif sim['run_type'] == 'batch':
                    unmixed, alg['state'] = alg['func'](sim['mixed'], alg['state'], alg.get('options'))
                else:
                    raise ValueError('unknown run_type={}'.format(sim['run_type']))
                unmixed = normalize_rowwise(unmixed)
                # play(unmixed[0] * 10000)
                alg['unmixed'] = unmixed

                # 5.2 Save data to wav files
                dir_alg = alg_create_folders(alg, dir_sim_unmixed)
                for file_name, s in zip(data_set['file_names'], unmixed):
                    write("{}/{}".format(dir_alg, file_name), data_set['fs'], np.float32(s))

                # 5.3 Compute metrics
                alg['metrics'] = {data_set['type']: evaluate(S, sim['filtered'], unmixed)}

            # delete temporary "mixed" array form dict
            del sim['mixed']

            # Create plots for this sim
            plot_sim_data_set_metrics(sim, data_set, dir_sim_plots)

    # Collect all metrics into new dictionary, display in in console with correct view and plot the results in folder
    rew_sims = rework_dict(sims)
    print_results(rew_sims)
    plot_metrics(rew_sims, dir_plots)


def evaluate(original: np.ndarray, filtered: np.ndarray, unmixed: np.ndarray) -> dict:
    ref = np.moveaxis(filtered, 1, 2)
    Ns = np.minimum(unmixed.shape[1], ref.shape[1])
    Sn = np.minimum(unmixed.shape[0], ref.shape[0])
    SDR, SIR, SAR, P = bss_eval_sources(ref[: Sn, :Ns, 0], unmixed[: Sn, :Ns])
    # TODO: RMSE was removed because of Singular Matrix error, uncomment for check
    return {'SDR': SDR, 'SIR': SIR, 'SAR': SAR, 'P': P, 'RMSE': rmse(original, unmixed)}
    # return {'SDR': SDR, 'SIR': SIR, 'SAR': SAR, 'P': P}


def create_folders() -> Tuple[str, str]:
    # Create sim folder
    dir_sims = "Sims"
    if not os.path.isdir(dir_sims):
        os.mkdir(dir_sims)

    # Create folder for plots
    dir_plots = "{}/plots".format(dir_sims)
    if not os.path.isdir(dir_plots):
        os.mkdir(dir_plots)

    return dir_sims, dir_plots


def sim_create_folders(sim: dict, dir_sims: str) -> Tuple[str, str, str]:
    dir_sim = "{}/{}_{}".format(dir_sims, sim['name'], datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.isdir(dir_sim):
        os.mkdir(dir_sim)

    dir_sim_unmixed = "{}/unmixed".format(dir_sim)
    if not os.path.isdir(dir_sim_unmixed):
        os.mkdir(dir_sim_unmixed)

    dir_sim_plots = "{}/plots".format(dir_sim)
    if not os.path.isdir(dir_sim_plots):
        os.mkdir(dir_sim_plots)

    return dir_sim, dir_sim_unmixed, dir_sim_plots


def alg_create_folders(alg: dict, dir_sim: str) -> str:
    dir_alg = "{}/{}".format(dir_sim, alg['name'])
    if not os.path.isdir(dir_alg):
        os.mkdir(dir_alg)

    return dir_alg


if __name__ == "__main__":
    main()
