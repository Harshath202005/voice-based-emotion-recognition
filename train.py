import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder

# Load preprocessed data
data = np.load("data.npy")
labels = np.load("labels.npy")

# Reshape for CNN (add channel dimension: grayscale=1, RGB=3)
data = data.reshape(data.shape[0], 128, 128, 1)

# Convert labels to numbers (e.g., "happy" -> 0, "sad" -> 1)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(np.unique(labels_encoded)), activation='softmax')  # Output layer
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(data, labels_encoded, epochs=10, validation_split=0.2)
model.save("emotion_model.h5")  # Save the trained model

from sklearn.metrics import classification_report

# Get predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print performance per emotion
print(classification_report(y_test, y_pred_classes, target_names=emotions))