import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model import build_model
from utils import extract_mfcc

DATASET_DIR = 'RAVDESS'
emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

X, y = [], []
for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            label = emotion_map.get(file.split("-")[2])
            if label:
                mfcc = extract_mfcc(path)
                X.append(mfcc)
                y.append(label)

X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
np.save('label_classes.npy', le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

model = build_model(X_train.shape[1:], len(np.unique(y_encoded)))
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
model.save('emotion_model.h5')