import pyaudio
import time
import numpy as np


def callback(in_data, frame_count, time_info, flag):
    global fulldata
    audio_data = np.fromstring(in_data, dtype=np.float32)
    fulldata = np.append(fulldata, audio_data)
    return audio_data, pyaudio.paContinue

def Recorder():
    global fulldata


    p = pyaudio.PyAudio()
    fulldata = np.array([])
    stream = p.open(format=pyaudio.paFloat32,
                    rate=44100,
                    frames_per_buffer=2048,
                    channels=1,
                    input=True,
                    input_device_index=device_idx(),
                    stream_callback=callback)

    stream.start_stream()

    while stream.is_active():
        time.sleep(3)
        stream.stop_stream()
    stream.close()
    p.terminate()


def device_idx():
    import sys

    p = pyaudio.PyAudio()
    device_idx = None
    print('\033[96mMicrophone is :\033[0m')
    for i in range(p.get_device_count()):
        if "miniDSP" in p.get_device_info_by_index(i)['name']:
            device_idx = i
            print("\t", p.get_device_info_by_index(device_idx)['name'])
            return device_idx
        #TODO:Small fix when don't have the miniDSP. Change the name below for correct work!
        elif "HD Webcam" in p.get_device_info_by_index(i)['name']:
            device_idx = i
            print("\t", p.get_device_info_by_index(device_idx)['name'])
            return device_idx

    if device_idx == None:
        print("\033[31m {}".format('Error : No Microphone'))
        sys.exit()

    return None
