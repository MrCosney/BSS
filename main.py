from Player import *
from setups import *
from Model import mix
from Normalizer import *
from plots import *
from scipy.io.wavfile import write
from mir_eval.separation import bss_eval_sources

from datetime import datetime
from typing import Tuple
import numpy.linalg as nl


def main():
    SDRR, SARR, SIRR, sims, data_sets = setups()

    dir_sims, dir_plots = create_folders()

    # Iterate over simulations
    for sim in sims:
        print('\033[35mSimulation \'{}\' started\033[0m'.format(sim['name']))
        # Create folders to save data
        dir_sim, dir_sim_filtered, dir_sim_mixed, dir_sim_unmixed, dir_sim_plots, dir_sim_box_plots = sim_create_folders(sim, dir_sims)

        # 1. Load source signals
        for data_set in sim['data_sets']:
            print('\t\033[35mDataSet \'{}\' \033[0m'.format(data_set['name']))
            S = []
            for si, wav in enumerate(data_set['data']):
                if si >= sim['sources']:
                    break
                if type(wav) == str:
                    S.append(load_wav(wav, data_set['fs']))
                else:
                    S.append(wav)
            # 2. Normalize & format source signals
            S = form_source_matrix(S)
            S = normalize_rowwise(np.array(S))
            data_set['audio_duration'] = round(S.shape[1] / data_set['fs'], 1)
            # 3. Perform environment simulation (mix signals)
            print('\t\t\033[35mMixing signals...\033[0m')
            filtered, mixed, sim = mix(S, sim, data_set)

            # 4. Normalize filtered & mixed arrays
            mixed = normalize(mixed)
            for f in filtered:
                for i in range(f.shape[0]):
                    f[i] = normalize(f[i])
            sim['filtered'] = filtered
            sim['mixed'] = mixed

            # 4.1. Save filtered & mixed plots
            pr = "{}_".format(data_set['name'])
            plot_original(S, dir_sim_mixed, pr, S.shape[1])
            plot_filtered(filtered, dir_sim_filtered, pr, S.shape[1])
            plot_mixed(mixed, dir_sim_mixed, pr, S.shape[1])

            # 4.2. Save filtered & mixed to wav
            for file_name, f in zip(data_set['file_names'], filtered):
                for mi, m in enumerate(f):
                    write("{}/{}_{}_mic_{}.wav".format(dir_sim_filtered, data_set['name'], file_name, mi), data_set['fs'], np.float32(m))

            for mi, m in enumerate(mixed):
                write("{}/{}_mic_{}.wav".format(dir_sim_mixed, data_set['name'], mi), data_set['fs'], np.float32(m))

            # 4.3. Create list of chunks (online version only)
            if sim['run_type'] == 'online':
                mixed_queue = rework_conv(mixed, sim)
            else:
                mixed_queue = []

            SDR_temp = []
            SIR_temp = []
            SAR_temp = []
            # 5. Run algorithms
            print('\t\t\033[35mSeparating...\033[0m')
            for alg in sim['algs']:

                if alg['name'].find('ILRMA') == 0 and data_set['name'] == 'Gen Signals':
                    print('Warning: artificially generated signals are not used with ILRMA, skipping...')
                    continue

                print("\t\t\tSeparation by {} in simulation {} with chunk_size={} ..."
                      .format(alg['name'], sim['name'], sim['chunk_size'] if 'chunk_size' in sim else '-'))

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
                    alg['state'] = {}
                else:
                    raise ValueError('unknown run_type={}'.format(sim['run_type']))
                unmixed = normalize_rowwise(unmixed)
                # play(unmixed[0] * 10000)
                alg['unmixed'] = unmixed

                # 5.2 Save data to wav files
                dir_alg = alg_create_folders(alg, dir_sim_unmixed)
                for file_name, s in zip(data_set['file_names'], unmixed):
                    write("{}/{}_{}.wav".format(dir_alg, data_set['name'], file_name), data_set['fs'], np.float32(s))

                # 5.3 Compute metrics
                alg['metrics'] = {data_set['name']: evaluate(S, sim['filtered'], unmixed)}
                SDR_temp.append(alg['metrics'][data_set['name']]['SDR'])
                SIR_temp.append(alg['metrics'][data_set['name']]['SIR'])
                SAR_temp.append(alg['metrics'][data_set['name']]['SAR'])

            # delete temporary "mixed" array form dict
            del sim['mixed']
            SDRR.append(SDR_temp)
            SARR.append(SAR_temp)
            SIRR.append(SIR_temp)
            # Create plots for this sim
            # plot_sim_data_set_metrics(sim, data_set, dir_sim_plots)

        plot_boxes(SDRR, SARR, SIRR, sim['name'], dir_sim_box_plots)
        print('\033[35mSimulation \'{}\' finished\033[0m'.format(sim['name']))

    print('\033[35mSaving stuff...\033[0m')
    # Collect all metrics into new dictionary, display in in console with correct view and plot the results in folder
    rew_sims = rework_dict(sims)
    print_results(rew_sims)
    # plot_metrics(rew_sims, dir_plots)
    print('\033[35mAll done.\033[0m')
    print(SDRR)


def evaluate(original: np.ndarray, filtered: np.ndarray, unmixed: np.ndarray) -> dict:
    ref = np.moveaxis(filtered, 1, 2)
    Ns = np.minimum(unmixed.shape[1], ref.shape[1])
    # Sn = np.minimum(unmixed.shape[0], ref.shape[0])
    SDR, SIR, SAR, P = bss_eval_sources(ref[:, :Ns, 0], unmixed[:, :Ns])
    # return {'SDR': SDR, 'SIR': SIR, 'SAR': SAR, 'P': P, 'RMSE': rmse(original, unmixed)}
    return {'SDR': np.round(np.mean(SDR), 2),
            'SIR': np.round(np.mean(SIR), 2),
            'SAR': np.round(np.mean(SAR), 2),
            'P': P}


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


def sim_create_folders(sim: dict, dir_sims: str) -> Tuple[str, str, str, str, str, str]:
    dir_sim = "{}/{}_{}".format(dir_sims, sim['name'], datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    # dir_sim = "{}/{}".format(dir_sims, sim['name'])  # without date - easier for development
    if not os.path.isdir(dir_sim):
        os.mkdir(dir_sim)

    dir_sim_unmixed = "{}/unmixed".format(dir_sim)
    if not os.path.isdir(dir_sim_unmixed):
        os.mkdir(dir_sim_unmixed)

    dir_sim_filtered = "{}/filtered".format(dir_sim)
    if not os.path.isdir(dir_sim_filtered):
        os.mkdir(dir_sim_filtered)

    dir_sim_mixed = "{}/mixed".format(dir_sim)
    if not os.path.isdir(dir_sim_mixed):
        os.mkdir(dir_sim_mixed)

    dir_sim_plots = "{}/plots".format(dir_sim)
    if not os.path.isdir(dir_sim_plots):
        os.mkdir(dir_sim_plots)

    dir_sim_box_plots = "{}/box_plots".format(dir_sim)
    if not os.path.isdir(dir_sim_box_plots):
        os.mkdir(dir_sim_box_plots)

    return dir_sim, dir_sim_filtered, dir_sim_mixed, dir_sim_unmixed, dir_sim_plots, dir_sim_box_plots


def alg_create_folders(alg: dict, dir_sim: str) -> str:
    dir_alg = "{}/{}".format(dir_sim, alg['name'])
    if not os.path.isdir(dir_alg):
        os.mkdir(dir_alg)

    return dir_alg


if __name__ == "__main__":
    main()
