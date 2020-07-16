#import the pyplot and wavfile modules
from scipy.io import wavfile
import numpy as np
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import librosa
import scipy
from sklearn import preprocessing
from pydub import AudioSegment
import os

class audio:
    def __init__(self,file):
        self.path  = file
        self.name = file.split('/')[-1].split('.')[0]
        self.signalData,self.samplingFrequency  = librosa.load(self.path)
        self.duration = librosa.get_duration(filename=self.path)
    def plot_wav(self):
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(self.signalData, sr=self.samplingFrequency)
    def plot_spectrogram(self):
        X = librosa.stft(self.signalData)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        plt.title(self.name.split('.')[0])
        librosa.display.specshow(Xdb, sr=self.samplingFrequency, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')





def freq_plot(self):
    X = scipy.fft(self.signalData)
    X_mag = np.absolute(X)
    f = np.linspace(0, self.samplingFrequency, len(X_mag))
    plt.figure(figsize=(14, 5))
    plt.plot(f, X_mag) # magnitude spectrum
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')


def plot_wav(self,save = False):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(self.signalData, sr=self.samplingFrequency)
    if save == True:
        plt.savefig('wavfile-'+self.name.split('.')[0]+'.png')

def plot_spectrogram(self,save = False):
    plt.figure(figsize=(14, 5))
    X = librosa.stft(self.signalData)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    plt.title(self.name.split('.')[0])
    librosa.display.specshow(Xdb, sr=self.samplingFrequency, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    if save == True:
        plt.savefig('spectogram-'+self.name.split('.')[0]+'.png')

def plot_spectrogram_limit(self,limits = (0,10000),save=False):
    plt.figure(figsize=(14, 10))
    # Plot the signal read from wav file
    plt.title(self.name)
    t = np.linspace(0,self.duration,len(self.signalData))
    plt.specgram(self.signalData,Fs=self.samplingFrequency,cmap = 'inferno')
    plt.colorbar(format='%+2.0f hz')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.axis(ymin=limits[0], ymax=limits[1])
    plt.show()
    if save == True:
        plt.savefig('Sprectogramlimitedfrequency-'+self.name.split('.')[0]+'.png')

def rmse(self):
    hop_length = 256
    frame_length = 512
    energy = np.array([sum(abs(self.signalData[i:i+frame_length]**2)) for i in range(0, len(self.signalData), hop_length)
                    ])
    rmse = librosa.feature.rms(self.signalData, frame_length=frame_length, hop_length=hop_length, center=True)
    frames = range(len(energy))
    t = librosa.frames_to_time(frames, sr=self.samplingFrequency, hop_length=hop_length)
