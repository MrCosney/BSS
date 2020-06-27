import pyaudio
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from setups import speakers_device_idx
from RecorderClass import Recorder
import threading

def play(audio: np.array, device_idx):
    p = pyaudio.PyAudio()
    stream = p.open(format=
                    pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    output=True,
                    output_device_index=device_idx)
    # read data (based on the chunk size)
    audio = np.clip(audio.transpose(), -2 ** 15, 2 ** 15 - 1)
    sound = (audio.astype(np.int16).tostring())

    print('\033[34m Player' + str(device_idx) + ':\033[0m', ' Reproduce audio...')
    stream.write(sound)
    stream.stop_stream()
    stream.close()
    print('\033[34m Player' + str(device_idx) + ':\033[0m', ' Done.')
    p.terminate()


def load_wav(filename, samplerate=44100):

    rate, data = wavfile.read(filename)
    if len(data.shape) > 1:
        data = data[:, 0] / 2 + data[:, 1] / 2

    # re-interpolate samplerate
    ratio = float(samplerate) / float(rate)
    data = resample(data, int(len(data) * ratio))

    return data

def play_and_record(X:np.array, data_set:dict, sim:dict):
    '''Play audio data on Speakers and Record via MiniDSP'''
    import sys
    # TODO: Maybe rewrite with Multiprocessing Pool class, I tried but it wait the thread for some reason (Commented version on the bottom of this page)
    vol_gain = 1000
    idx = speakers_device_idx()

    X = X * vol_gain
    recorder = Recorder(kwargs=({'fs': data_set['fs'],
                                 'chunk_size': sim['chunk_size'],
                                 'audio_duration': data_set['audio_duration'],
                                 'microphones': sim['microphones']}))
    rec = threading.Thread(target=recorder._record)
    z = X.shape[0]
    if len(idx) == 3:
        s1 = threading.Thread(target=play, args=(X[0], idx[0]))
        s2 = threading.Thread(target=play, args=(X[1], idx[1]))
        s3 = threading.Thread(target=play, args=(X[2], idx[2]))
        # start threads and wait till last speaker is done
        rec.start()
        s1.start()
        s2.start()
        s3.start()

    elif len(idx) == 2:
        s1 = threading.Thread(target=play, args=(X[0], idx[0]))
        s2 = threading.Thread(target=play, args=(X[1], idx[1]))
        rec.start()
        s1.start()
        s2.start()
    else:
        s1 = threading.Thread(target=play, args=(X[0], idx[0]))
        rec.start()
        s1.start()

    rec.join()

    return recorder._data