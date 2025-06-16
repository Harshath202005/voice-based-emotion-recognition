import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model and label encoder
model = load_model("emotion_model.h5")
labels = np.load("labels.npy")
label_encoder = LabelEncoder().fit(labels)

def predict_emotion(RAVDESS):
    # Preprocess the audio
    signal, sr = librosa.load(r"RAVDESS\Actor_01\03-01-01-01-01-02-01.wav", sr=22050)
    spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_db = np.resize(spectrogram_db, (128, 128, 1))
    
    # Predict
    prediction = model.predict(np.array([spectrogram_db]))
    predicted_emotion = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_emotion[0]

# Test with your audio file
print("Predicted Emotion:", predict_emotion("your_test_audio.wav"))