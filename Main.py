from Algorithms import *
from plots import *
from Normalizer import *
from scipy.io.wavfile import write
from Room import *
from Olegs import shullers_method
from SourceData import exctract


def main():
    mix_folder = "Audio/Mixed/"
    unmix_folder = "Audio/Unmixed/"
    unmix_folder_icaa = "Audio/Unmixed_Icaa/"
    unmix_folder_sh = "Audio/Unmixed_Shuller/"
    file_names = np.array(["Kunkka.wav", "Ench.wav", "Timber.wav"])
    rate, source_data, source_sig = exctract()
    '''Normalize input data'''
    normalization(source_data)
    '''Mixing input data'''
    mix_audio = mix(source_data[:2])  # mix 2 audio  sources for shullers method
    #mix_audio3 = mix(source_data)
    mix_signal_sh = mix(source_sig[:2])  # mix 2 signal sources for shullers method
    mix_room_audio = makeroom(rate, source_data)  # mix in simulation room for 3 sources
    mix_signals = mix(source_sig)
    '''Make Unmixing by Jade Algorithm'''
    JU = jade_unmix(mix_room_audio)
    JU_s = jade_unmix(mix_signals)
    '''Make Unmixing by ICAA Algorithm'''
    icaa = Fast(mix_signals.T)
    icaa_audio = Fast(mix_room_audio.T)
    '''Make Unmixing by Shuller Algorithm'''
    Shuller = shullers_method(mix_audio.T)
    Shuller_sig = shullers_method(mix_signal_sh.T)
    '''Swap lines and mirroring to correct plots view'''
    swap(JU, JU_s, icaa_audio, icaa)
    '''Normalize unmixing data '''
    JU = normalization(JU)
    JU_s = normalization(JU_s)
    icaa = normalization(icaa)
    icaa_audio = normalization(icaa_audio)
    Shuller = normalization(Shuller)
    mix_audio = normalization(mix_audio)
    mix_room_audio = normalization(mix_room_audio)
    Shuller_sig = normalization(Shuller_sig)
    '''Compute RMSE'''
    rms = rmse(source_data, JU)
    rms_sig = rmse(source_sig, JU_s)
    icaa_rms_sig = rmse(source_sig, icaa)
    icaa_rms_audio = rmse(source_data, icaa_audio)
    sh_rms = rmse(source_data[:2], Shuller)
    sh_rms_sig = rmse(source_sig[:2], Shuller_sig)
    '''Write data into .wav and plot graphs'''
    plots(source_data, mix_room_audio, JU, rms, "Fig/Audio unmixing JADE.jpeg", th=0.4, th1=0.4)
    plots(source_data, mix_room_audio, icaa_audio, icaa_rms_audio, "Fig/Audio unmixing ICAA.jpeg", th=0.4, th1=0.4)
    plots(source_sig, mix_signals, JU_s, rms_sig, "Fig/Gen.Signal unmixing JADE.jpeg", th=1)
    plots(source_sig, mix_signals, icaa, icaa_rms_sig, "Fig/Gen.Signal unmixing ICAA.jpeg", th=1)
    plots(source_data[:2], mix_audio, Shuller, sh_rms, "Fig/Audio unmixing Shuller.jpeg", th=0.4, th1=0.4)
    plots(source_sig[:2], mix_signal_sh, Shuller_sig, sh_rms_sig, "Fig/Gen.Signal unmixing Shuller.jpeg", th=0.4, th1=0.4)
    '''Write data into .wav'''
    freq = 44100
    for i in range(3):
        write("".join((mix_folder, file_names[i])), freq, np.float32(mix_room_audio[i]))
        write("".join((unmix_folder, file_names[i])), freq, np.float32(JU[i]))
        write("".join((unmix_folder_icaa, file_names[i])), freq, np.float32(icaa_audio[i]))
        if i < 2:
            write("".join((unmix_folder_sh, file_names[i])), freq, np.float32(Shuller[i]))
    print("_________________________________________")
    print('Quick Readme. \n If you want to play data, use - play(data[i] * 5000)\n All Fig. and .wav audio inside the folders\n')
    print("Enjoy =:")
    print("_________________________________________")
if __name__ == "__main__":
    main()
