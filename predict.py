import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf
from utils import extract_mfcc

model = tf.keras.models.load_model('emotion_model.h5')
label_classes = np.load('label_classes.npy')

def record_audio(duration=3, sr=22050):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("✅ Recording complete.")
    return audio.flatten(), sr

def predict_emotion():
    audio, sr = record_audio()
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    pad_width = 300 - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :300]
    mfcc = mfcc.T[np.newaxis, :, :]
    prediction = model.predict(mfcc)
    emotion = label_classes[np.argmax(prediction)]
    print(f"🧠 Predicted Emotion: {emotion}")

if __name__ == "__main__":
    predict_emotion()