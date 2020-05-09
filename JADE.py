from shogun import RealFeatures
from shogun import Jade
from plots import plots
from plots2 import plots2
from Normalizer import *
from scipy.io.wavfile import write
import timeit
from scipy import signal
from sklearn.decomposition import FastICA
from Room import *
from Olegs import shullers_method

def mix_2(s1, s2):
	ts1 = np.copy(s1)
	ts2 = np.copy(s2)
	length = max(len(ts1), len(ts2))
	ts1.resize((length, 1), refcheck=False)
	ts2.resize((length, 1), refcheck=False)

	S = (np.c_[ts1, ts2]).T
	'''Create mix matrix'''
	A = np.array([[1, 0.5],
				  [0.5, 1],
	])
	x = np.dot(A, S)
	return x

def mix(s1, s2, s3):
	length = max(len(s1), len(s2), len(s3))
	s1.resize((length, 1), refcheck=False)
	s2.resize((length, 1), refcheck=False)
	s3.resize((length, 1), refcheck=False)

	S = (np.c_[s1, s2, s3]).T
	'''Adding interference to signals'''
	if len(s1) == 10000:
		S += 0.1 * np.random.normal(size=S.shape)
	'''Create mix matrix'''
	A = np.array([[1, 0.5, 0.5],
				  [0.5, 1, 0.5],
				  [0.5, 0.5, 1]])

	x = np.dot(A, S)
	return x

def jade_unmix(mix_audio):
	a = timeit.default_timer()
	mixed_signals = RealFeatures(mix_audio.astype(np.float64))
	jade = Jade()
	signals = jade.apply(mixed_signals)
	JUnmixAudio = signals.get_feature_matrix()
	Mix_matrix = jade.get_mixing_matrix()
	Mix_matrix / Mix_matrix.sum(axis=0)
	t = round(timeit.default_timer() - a, 5)
	print("Estimated Mixing Matrix is: ")
	print(Mix_matrix)
	return JUnmixAudio, t

def Fast(mix_audio):
	ica = FastICA(n_components=3)
	S_ = ica.fit_transform(mix_audio)
	Mix_matrix = ica.mixing_
	# Mix_matrix / Mix_matrix.sum(axis=0)
	print("Estimated Mixing Matrix is: ")
	print(Mix_matrix)
	return S_.T

def main():
	freq = 44100
	mix_folder = "Audio/Mixed/"
	unmix_folder = "Audio/Unmixed/"
	unmix_folder_icaa = "Audio/Unmixed_Icaa/"
	file_names = np.array(["Kunkka.wav", "Ench.wav", "Timber.wav"])
	audio_1 = "Audio/Original/Kunkka.wav"
	audio_2 = "Audio/Original/Ench.wav"
	audio_3 = "Audio/Original/Timber.wav"
	audio_4 = "Audio/Original/Oleg.wav"
	'''Extract data from wav. files'''
	rate1, data1 = load_wav(audio_1, freq)
	rate2, data2 = load_wav(audio_2, freq)
	rate3, data3 = load_wav(audio_3, freq)
	rate4, data4 = load_wav(audio_4, freq)
	'''Create simple signals.'''
	np.random.seed(0)
	n_samples = 10000
	time = np.linspace(0, 8, n_samples)
	sig1 = np.sin(2 * time)						#Simple sin signal
	sig2 = np.sign(np.sin(3 * time)) 			#Simple square signal
	sig3 = signal.sawtooth(2 * np.pi * time)	#Simple saw signal
	'''Normalize input data'''
	mix_audio = mix_2(data1, data2)  # mix 2 sources for shullers method todo: fix it in 1 func
	mix_signal_sh = mix_2(sig1, sig2)
	data1 = normalization(data1)
	data2 = normalization(data2)
	data3 = normalization(data3)
	'''Mixing input data'''
	mix_room_audio = makeroom(rate1, data1, data2, data3)	#mix in simulation room for 3 sources
	mix_signals = mix(sig1, sig2, sig3)
	'''Make Unmixing by Jade Algorithm'''
	JU, t = jade_unmix(mix_room_audio)
	JU_s, t = jade_unmix(mix_signals)
	'''Make Unmixing by ICAA Algorithm'''
	icaa = Fast(mix_signals.T)
	icaa_audio = Fast(mix_room_audio.T)
	'''Make Unmixing by Shuller Algorithm'''
	Shuller = shullers_method(mix_audio.T)
	Shuller_sig = shullers_method(mix_signal_sh.T)
	'''Swap lines to correct view'''
	#Todo: Remove all this stuff from here
	swap_lines(JU, 0, 1)
	swap_lines(JU_s, 0, 1)
	swap_lines(icaa_audio, 0, 2)
	swap_lines(icaa_audio, 0, 1)
	swap_lines(icaa, 0, 2)
	swap_lines(icaa, 0, 1)
	'''Mirroring'''
	JU[1] *= -1
	JU[2] *= -1
	JU_s[1] *= -1
	JU_s[2] *= -1
	icaa_audio[2] *= -1
	'''Normalize data and gets RMSE b/w input and unmixed data and reverse order'''
	for i in range(JU.shape[0]):
		JU[i] = normalization2(JU[i])
		JU_s[i] = normalization2(JU_s[i])
		icaa[i] = normalization2(icaa[i])
		icaa_audio[i] = normalization2(icaa_audio[i])
		if i < 2:
			Shuller[i] = normalization2(Shuller[i])
			mix_audio[i] = normalization2(mix_audio[i])
	rms = rmse(data1, data2, data3, JU)
	rms_sig = rmse_s(sig1, sig2, sig3, JU_s)
	icaa_rms_sig = rmse_s(sig1, sig2, sig3, icaa)
	icaa_rms_audio = rmse(data1, data2, data3, icaa_audio)
	#sh_rms = rmse2(data1, data2, Shuller)
	'''Write data into .wav and plot graphs'''
	#Todo:organize plots part
	plots(data1, data2, data3, mix_room_audio, JU, rms, "Fig/Audio unmixing JADE.jpeg", th=0.4, th1=0.4)
	plots(sig1, sig2, sig3, mix_signals, JU_s, rms_sig, "Fig/Gen.Signal unmixing JADE.jpeg", th=1)
	plots(sig1, sig2, sig3, mix_signals, icaa, icaa_rms_sig, "Fig/Gen.Signal unmixing ICAA.jpeg", th=1)
	plots(data1, data2, data3, mix_room_audio, icaa_audio, icaa_rms_audio, "Fig/Audio unmixing ICAA.jpeg", th=0.4, th1=0.4)
	plots2(data1, data2, mix_audio, Shuller, "Fig/Audio unmixing Shuller.jpeg", th=0.4, th1=0.4)
	plots2(sig1, sig2, mix_signal_sh, Shuller_sig, "Fig/Audio unmixing Shuller signals.jpeg", th=0.4, th1=0.4)
	'''Write data into .wav'''
	for i in range(3):
		write("".join((mix_folder, file_names[i])), freq, np.float32(mix_room_audio[i]))
		write("".join((unmix_folder, file_names[i])), freq, np.float32(JU[i]))
		write("".join((unmix_folder_icaa, file_names[i])), freq, np.float32(icaa_audio[i]))
if __name__ == "__main__":
	main()
