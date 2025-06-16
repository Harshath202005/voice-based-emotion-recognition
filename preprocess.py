import librosa
import numpy as np
import os

emotions = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

data = []
labels = []

for root, dirs, files in os.walk("RAVDESS"):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            emotion_code = file.split("-")[2]
            emotion = emotions[emotion_code]
            
            # Load audio and extract spectrogram
            signal, sr = librosa.load(file_path, sr=22050)
            spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            
            # Resize to 128x128 (CNNs need fixed-size inputs)
            spectrogram_db = np.resize(spectrogram_db, (128, 128))
            
            data.append(spectrogram_db)
            labels.append(emotion)

# Save to files (to avoid reprocessing later)
np.save("data.npy", np.array(data))
np.save("labels.npy", np.array(labels))