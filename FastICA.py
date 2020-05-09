import numpy as np
from sklearn.decomposition import FastICA
from plots import plots
import matplotlib.pyplot as plt

def Fast(mix_audio):
    mix_audio = np.reshape(mix_audio, (len(mix_audio[0]), mix_audio.shape[0]))
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(mix_audio)
    Mix_matrix = ica.mixing_

    # Mix_matrix / Mix_matrix.sum(axis=0)
    print("Estimated Mixing Matrix is: ")
    print(Mix_matrix)
    return S_




def fastica(sig1, sig2, sig3, rms_sig):

    S = np.c_[sig1, sig2, sig3]
    #S += 0.2 * np.random.normal(size=S.shape)

    S /= S.std(axis=0)
    # Mix data
    A = np.array([[1, 1, 1],
                  [0.5, 2, 1.0],
                  [1.5, 1.0, 2.0]])  # Mixing matrix
    X = np.dot(S, A.T)  # Generate observations

    # Compute ICA
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix
    # Plot results
    S_ = S_.T
    for i in range(3):
        plt.plot(S_[i], color='red')
        plt.show()

    plots(sig1, sig2, sig3, X, S_, rms_sig, "Fig/Gen.Signal unmixing.jpeg", th=0.1)


    plt.figure()

    models = [S, X, S_]
    names = ['True Sources',
             'Mixed signal)',
             'Recovered']
    colors = ['black', 'steelblue', 'orange']

    for ii, (model, name) in enumerate(zip(models, names), 1):
        plt.subplot(3, 1, ii)
        plt.title(name)
        for sig, color in zip(model.T, colors):
            plt.plot(sig, color=color)

    plt.tight_layout()
    plt.show()