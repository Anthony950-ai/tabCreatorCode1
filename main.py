import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

with open("C:/Users/Mario/Desktop/Audios/audios.txt" , 'r') as file:
    content = file.read()
    words = content.split()


n_fft = 1024  # Window size
hop_length = 256  # Hop size (overlap)

#list that holds all frequencies
allFrequencies = []

for af in words:
    # Load an audio file
    audio_file = "C:/Users/Mario/Desktop/Audios/" + af
    y, sr = librosa.load(audio_file)

    # Calculate the STFT
    n_fft = 1024  # Window size
    hop_length = 256  # Hop size (overlap)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Calculate the magnitude spectrogram
    magnitude_spec = np.abs(D)

    # Calculate the center frequencies manually
    freqs = np.fft.fftfreq(n_fft, 1 / sr)
    freqs = freqs[:n_fft // 2]  # Keep only positive frequencies

    # Define the target time in seconds (e.g., 0.5 seconds)
    target_time = 1.0

    # Find the index of the time frame closest to the target time
    target_frame_index = int(np.round(target_time * sr / hop_length))

    # Get the frequencies and magnitudes for the target time frame
    target_frame = magnitude_spec[:, target_frame_index]

    # Find the 5 most prevalent frequencies at the target time frame
    num_top_frequencies = 7
    top_indices = np.argsort(target_frame)[-num_top_frequencies:]
    top_frequencies = freqs[top_indices]
    top_magnitudes = target_frame[top_indices]


    # Print the n most prevalent frequencies and their magnitudes at the target time frame
    print("For " + af)
    for i in range(num_top_frequencies):
        print(f"Top Frequency {num_top_frequencies - i}: {top_frequencies[i]} Hz (Magnitude: {top_magnitudes[i]})")
        allFrequencies.append(top_frequencies[i])

    print("\n")
with open('data.txt', 'w') as file:
    for i in range(len(allFrequencies)):
        if(i % num_top_frequencies == 0) or (i == 0):
            file.write(str(allFrequencies[i]) + ",")
        else:
            file.write(str(allFrequencies[i]) + "\n")




    #plt.figure(figsize=(10, 6))
    #librosa.display.specshow(librosa.amplitude_to_db(magnitude_spec, ref=np.max), sr=sr, hop_length=hop_length,
    #                      x_axis='time', y_axis='hz')
    # plt.colorbar(format='%+2.0f dB')
    #plt.title('Spectrogram')
    # plt.show()

    #input("continue")





