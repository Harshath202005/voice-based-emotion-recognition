import librosa
import matplotlib.pyplot as plt

# Load an audio file (update the path)
file_path = "RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"
signal, sr = librosa.load(file_path, sr=22050)  # sr = sample rate

# Plot the raw audio waveform
plt.figure(figsize=(10, 4))
plt.plot(signal)
plt.title("Audio Waveform")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.show()