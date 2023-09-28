import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.interpolate import interp1d

with open("D:/Capstone/audios.txt" , 'r') as file:
    content = file.read()
    words = content.split()
#test
for af in words:
    # Load an audio file
    audio_file = "D:/Capstone/" + af
    y, sr = librosa.load(audio_file)

    D = librosa.stft(y)

    spectrogram = librosa.amplitude_to_db(abs(D), ref = np.max)

    timeList = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    frequencyList = []
    for t in timeList:
        time_index = librosa.time_to_frames(t, sr=sr)
        frequencies = librosa.fft_frequencies(sr=sr)
        frequency_at_time = frequencies[np.argmax(spectrogram[:, time_index])]
        frequencyList.append(frequency_at_time)

    print (af)
    print(frequencyList)
    print()
    spectro = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram_db = librosa.power_to_db(spectro, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(spectrogram_db, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency Spectrogram' + af)
    plt.show()
    inpu = input("continue?")
