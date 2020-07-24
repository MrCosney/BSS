import pyaudio
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample


def play(audio: np.array, device_idx):
    p = pyaudio.PyAudio()
    stream = p.open(format=
                    pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    output=True,
                    output_device_index=device_idx)
    # read data (based on the chunk size)
    audio = np.clip(audio.transpose(), -2 ** 15, 2 ** 15 - 1)
    sound = (audio.astype(np.int16).tostring())

    print('\t\033[34m Player' + str(device_idx) + ':\033[0m', ' Reproduce audio...')
    stream.write(sound)
    stream.stop_stream()
    stream.close()
    print('\t\033[34m Player' + str(device_idx) + ':\033[0m', ' Done.')
    p.terminate()


def load_wav(filename, samplerate=44100):

    rate, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0] / 2 + data[:, 1] / 2

    # re-interpolate samplerate
    #ratio = float(samplerate) / float(rate)
    #data = resample(data, int(len(data) * ratio))

    return data
