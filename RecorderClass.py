import pyaudio
import numpy as np
import time


class Recorder:
    def __init__(self, **kwargs):
        self.__p = pyaudio.PyAudio()
        self.kwargs = kwargs['kwargs']
        self._rate = self.kwargs['fs'] if 'fs' in self.kwargs else 44100
        self._chunk_size = self.kwargs['chunk_size'] if 'chunk_size' in self.kwargs else 2048
        self._audio_duration = self.kwargs['audio_duration'] if 'audio_duration' in self.kwargs else 4
        self._channels = self.kwargs['microphones'] if 'microphones' in self.kwargs else 2
        self.device_idx = self.device_idx()
        self.__stream = None
        self.__callbackFlag = None
        self._data = None

    def _record(self):
        self._data = []
        print('\t \033[31mRecorder:\033[0m', ' Audio is Recording by ', self._channels, ' Microphones...')
        self.__callbackFlag = pyaudio.paContinue
        self.__stream = self.__p.open(format=pyaudio.paFloat32,
                                      rate=self._rate,
                                      frames_per_buffer=self._chunk_size,
                                      channels=self._channels,
                                      input=True,
                                      input_device_index=self.device_idx,
                                      stream_callback=self.__callback)

        # TODO: Make normal stop that depends on audio duration
        while self.__stream.is_active():
            time.sleep(self._audio_duration)
            self._stop_record()
        self.__stream.close()

        return self._data

    def _stop_record(self):
        self.__callbackFlag = pyaudio.paComplete
        self.__stream.stop_stream()
        print('\t\033[31m Recorder:\033[0m', ' Audio is Recorded.')

    def __callback(self, in_data, frame_count, time_info, status):
        in_data = np.fromstring(in_data, 'Float32').reshape((frame_count, self._channels)).transpose()
        self._data.append(in_data)
        return np.zeros((0, 0)), self.__callbackFlag

    def device_idx(self):
        """This method check if mic.array is connected or not and returns device index if it is connected"""
        import sys

        __device_idx = None
        print('\033[96mMicrophone is:\033[0m')
        for i in range(self.__p.get_device_count()):
            if "miniDSP" in self.__p.get_device_info_by_index(i)['name']:
                __device_idx = i
                print("\t", self.__p.get_device_info_by_index(i)['name'])
                return __device_idx

        if __device_idx is None:
            print("\033[31m{}".format('\tError : The MiniDsp is not connected!'))
            sys.exit()
        return None
