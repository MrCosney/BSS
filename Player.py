import pyaudio
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample


def play(audio):
    p = pyaudio.PyAudio()
    chunk = 1024
    stream = p.open(format=
                    pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True)
    # read data (based on the chunk size)
    audio = np.clip(audio.transpose(), -2 ** 15, 2 ** 15 - 1)
    sound = (audio.astype(np.int16).tostring())

    stream.write(sound)

    stream.stop_stream()
    stream.close()
    p.terminate()
    p.terminate()

def load_wav(filename, samplerate=44100):

    rate, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0] / 2 + data[:, 1] / 2

    # re-interpolate samplerate
    ratio = float(samplerate) / float(rate)
    data = resample(data, int(len(data) * ratio))

    return samplerate, data