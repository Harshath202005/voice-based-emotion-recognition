# voice-based-emotion-recognition
Voice Emotion Recognition System
An AI-powered system that classifies human emotions from voice recordings using deep learning and signal processing techniques.


📋 Project Overview
This project uses Convolutional Neural Networks (CNNs) to analyze voice recordings and classify them into 8 emotional categories:

😊 Happy

😢 Sad

😠 Angry

😨 Fearful

😐 Neutral

😲 Surprised

😌 Calm

🤢 Disgust

🚀 Features
Voice Analysis: Converts audio to mel spectrograms for visual pattern recognition

Deep Learning Model: CNN architecture optimized for audio classification

Web Interface: User-friendly Flask web application

Real-time Prediction: Instant emotion detection from uploaded audio files

Data Augmentation: Enhanced training with synthetic audio variations

📁 Project Structure
text
voice-emotion-recognition/
├── RAVDESS/                    # Dataset directory
├── static/                     # Web assets (CSS, JS)
├── templates/                  # HTML templates
│   └── index.html             # Main web interface
├── spectrograms.npy           # Preprocessed spectrograms
├── labels.npy                 # Emotion labels
├── emotion_model.h5           # Trained CNN model
├── app.py                     # Flask web application
├── create_spectrograms.py     # Audio preprocessing script
├── train_model.py             # Model training script
├── test_model.py              # Model testing script
└── requirements.txt           # Python dependencies
⚙️ Installation
Prerequisites
Python 3.8+

Anaconda (recommended)

Setup Steps
Clone the repository

bash
git clone https://github.com/your-username/voice-emotion-recognition.git
cd voice-emotion-recognition
Create and activate conda environment

bash
conda create -n emotion python=3.8
conda activate emotion
Install dependencies

bash
pip install -r requirements.txt
Download Dataset

Get RAVDESS dataset from Kaggle

Extract into RAVDESS/ folder in project directory

🏃‍♂️ Usage
1. Preprocess Audio Data
bash
python create_spectrograms.py
Converts audio files to spectrograms and saves as numpy arrays

2. Train the Model
bash
python train_model.py
Trains CNN model and saves as emotion_model.h5

3. Run Web Application
bash
python app.py
Starts Flask server at http://localhost:5000

4. Test with Custom Audio
bash
python test_model.py --audio path/to/your/audio.wav
🌐 Web Interface
Access the web app at http://localhost:5000:

Upload: Click "Choose File" to select a WAV audio file

Analyze: Click "Predict Emotion" to process

Results: View the predicted emotion category

https://via.placeholder.com/600x400?text=Voice+Emotion+Recognition+Web+App

🧠 Technical Details
Model Architecture
text
Input (128x128 spectrogram) → 
Conv2D (32 filters) → MaxPooling → 
Conv2D (64 filters) → MaxPooling → 
Flatten → Dense (128 units) → 
Output (8 emotions with softmax)
Audio Processing Pipeline
Load audio file (22.05 kHz sample rate)

Generate mel spectrogram

Convert to decibel scale

Resize to 128x128 pixels

Normalize and preprocess for CNN

📊 Performance
Training Accuracy: 85-90%

Validation Accuracy: 75-80%

Inference Time: < 2 seconds per audio file

Supported Formats: WAV, MP3 (auto-converted)

🔧 Customization
Add New Emotions
Edit the emotions dictionary in create_spectrograms.py:

python
emotions = {
    '01': 'neutral',
    '02': 'calm',
    # Add new mappings here
}
Modify Model Architecture
Edit train_model.py to change CNN layers:

python
model.add(Conv2D(64, (3, 3), activation='relu'))  # Add more layers
model.add(Dropout(0.3))  # Regularization
🤝 Contributing
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
RAVDESS Dataset: Ryerson Audio-Visual Database of Emotional Speech and Song

LibROSA: Audio and music processing library

TensorFlow: Machine learning framework

Flask: Web framework



Note: This project is for educational purposes. Always verify critical emotional assessments with qualified professionals.

<div align="center">
Made with ❤️ and 🎵

</div>
