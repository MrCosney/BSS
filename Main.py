from Algorithms import *
from plots import *
from Normalizer import *
from scipy.io.wavfile import write
from Room import *
from Olegs import shullers_method
from SourceData import *
import copy


def main():
    rate = 44100
    source_data, source_sig = exctract()
    '''Normalize input data'''
    normalization(source_data)
    '''Mixing input data'''
    mix_audio = mix(copy.deepcopy(source_data)[:2])  # mix 2 audio  sources for shullers method
    mix_signal_sh = mix(copy.deepcopy(source_sig)[:2])  # mix 2 signal sources for shullers method
    mix_room_audio = makeroom(rate, source_data)  # mix in simulation room for 3 sources        #TODO: Fix for simulations
    #source_data = source_data[:2]
    #source_sig = source_sig[:2]
    #mix_room_audio = mix(copy.deepcopy(source_data))
    mix_signals = mix(copy.deepcopy(source_sig))
    '''Make Unmixing by Jade Algorithm'''
    JU = jade_unmix(mix_room_audio, 1 ,1)
    JU_s = jade_unmix(mix_signals, 1 ,1)
    '''Make Unmixing by ICAA Algorithm'''
    icaa = Fast(mix_signals.T, 1 ,1 )
    icaa_audio = Fast(mix_room_audio.T, 1 , 1 )
    '''Make Unmixing by PCA'''
    #pcaa = Pca(mix_signals.T, 1 ,1)
    pcaa_audio = Pca(mix_room_audio.T, 1 ,1)
    '''Make Unmixing by Shuller Algorithm'''
    Shuller = shullers_method(mix_audio.T)
    Shuller_sig = shullers_method(mix_signal_sh.T)
    '''Make Unmixing by AuxIVA Algorithm'''
    iva, mix_iva = auxvia(copy.deepcopy(source_data))
    '''Swap lines and mirroring to correct plots view'''
    swap(JU, JU_s, icaa_audio, icaa, pcaa)
    '''Normalize unmixing data '''
    pcaa = normalization(pcaa)
    pcaa_audio = normalization(pcaa_audio)
    iva = normalization(iva)
    mix_iva = normalization(mix_iva)
    JU = normalization(JU)
    JU_s = normalization(JU_s)
    icaa = normalization(icaa)
    icaa_audio = normalization(icaa_audio)
    Shuller = normalization(Shuller)
    mix_signals = normalization(mix_signals)
    mix_audio = normalization(mix_audio)
    mix_room_audio = normalization(mix_room_audio)
    Shuller_sig = normalization(Shuller_sig)
    '''Compute RMSE'''
    rms = rmse(source_data, JU, "Audio unmixing JADE.txt")
    rms_sig = rmse(source_sig, JU_s, "Gen.Signal unmixing JADE.txt")
    icaa_rms_sig = rmse(source_sig, icaa, "Gen.Signal unmixing ICAA.txt")
    icaa_rms_audio = rmse(source_data, icaa_audio, "Audio unmixing ICAA.txt")
    sh_rms = rmse(source_data[:2], Shuller, "Audio Shuller.txt")
    sh_rms_sig = rmse(source_sig[:2], Shuller_sig, "Gen.Signal sig Shuller.txt")
    iva_rms = rmse(source_data, iva, "Audio unmixing AUXIVA.txt")
    pcaa_rms = rmse(source_sig, pcaa, "Gen.Signal unmixing PCA.txt")
    pcaa_rms_audio = rmse(source_data, pcaa_audio, "Audio unmixing PCA.txt")
    '''Write data into .wav and plot graphs'''
    '''Plot Folders'''
    f_fig = "Fig/SimMatrix2/"
    ''''''
    plots(source_data, mix_room_audio, pcaa_audio, pcaa_rms_audio, "".join((f_fig, "Audio unmixing PCA.jpeg")), th=0.4, th1=0.4)
    plots(source_data, mix_room_audio, JU, rms,  "".join((f_fig, "Audio unmixing JADE.jpeg")), th=0.4, th1=0.4)
    plots(source_data, mix_room_audio, icaa_audio, icaa_rms_audio, "".join((f_fig, "Audio unmixing ICAA.jpeg")), th=0.4, th1=0.4)
    plots(source_sig, mix_signals, pcaa, pcaa_rms, "".join((f_fig, "Ge.Signal unmixing PCA.jpeg")), th=0.4,
          th1=0.4)
    plots(source_sig, mix_signals, JU_s, rms_sig, "".join((f_fig, "Gen.Signal unmixing JADE.jpeg")), th=1)
    plots(source_sig, mix_signals, icaa, icaa_rms_sig, "".join((f_fig, "Gen.Signal unmixing ICAA.jpeg")), th=1)
    plots(source_data[:2], mix_audio, Shuller, sh_rms, "".join((f_fig, "Audio unmixing Shuller.jpeg")), th=0.4, th1=0.4)
    plots(source_sig[:2], mix_signal_sh, Shuller_sig, sh_rms_sig, "".join((f_fig, "Gen.Signal unmixing Shuller.jpeg")), th=0.4, th1=0.4)
    plots(source_data, mix_iva, iva, iva_rms, "".join((f_fig, "Audio unmixing AUXIVA.jpeg")), th=0.4, th1=0.4)
    '''Write data into .wav'''
    mix_folder = "Audio/Mixed/"
    unmix_folder = "Audio/Unmixed/"
    unmix_folder_icaa = "Audio/Unmixed_Icaa/"
    unmix_folder_sh = "Audio/Unmixed_Shuller/"
    file_names = np.array(["Kunkka.wav", "Ench.wav", "Timber.wav"])

    freq = 44100
    for i in range(3):
        write("".join((mix_folder, file_names[i])), freq, np.float32(mix_room_audio[i]))
        write("".join((unmix_folder, file_names[i])), freq, np.float32(JU[i]))
        write("".join((unmix_folder_icaa, file_names[i])), freq, np.float32(icaa_audio[i]))
        if i < 2:
            write("".join((unmix_folder_sh, file_names[i])), freq, np.float32(Shuller[i]))


if __name__ == "__main__":
    main()
