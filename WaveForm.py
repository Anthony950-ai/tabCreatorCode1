import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
audio_path = 'FullAudios/SameNoteSample.wav'
y, sr = librosa.load(audio_path)

# Calculate amplitude deltas
amplitude_deltas = [y[i] - y[i - 1] for i in range(1, len(y))]

# Calculate the time values manually
total_samples = len(y)
time_values = [i * (1 / sr) for i in range(1, total_samples)]
window_size = 1  # Adjust this value to control the degree of smoothing
amplitude_deltas = np.convolve(amplitude_deltas, np.ones(window_size)/window_size, mode='same')



noteCount = 0
netChange = 0
for i in range(1, total_samples - 1):
    netChange += amplitude_deltas[i]
    if netChange >= 0:
        noteCount += 1

print(amplitude_deltas)
print(noteCount)
print(netChange)
# Create a plot of amplitude deltas against time
plt.figure(figsize=(12, 6))
plt.plot(time_values, amplitude_deltas, label='Amplitude Deltas')
plt.title('Amplitude Deltas vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude Deltas')
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr, alpha=0.6, color='b')
plt.title("Waveform of Audio File")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()

# Show the plot
plt.show()