
# 🧠 Emotion Recognition Using SNN + GRU + Neural ODE

This is a hybrid emotion recognition model that combines:

- 🧬 Spiking Neural Networks (SNN) for biologically inspired processing
- 📈 Neural ODE for modeling continuous time transitions
- 🔁 GRU for sequential pattern learning

The system detects emotional states from speech audio using the RAVDESS dataset.

---

## 📁 Project Structure

| File              | Description |
|-------------------|-------------|
| `main.py`         | Final integrated research script |
| `train.py`        | Script to train the model from RAVDESS |
| `predict.py`      | Real-time microphone prediction |
| `model.py`        | Model definition using GRU + Conv1D |
| `utils.py`        | Audio feature extraction (MFCC) |
| `requirements.txt`| Required Python libraries |
| `README.md`       | Project overview |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py
```

### 3. Predict Emotion from Mic
```bash
python predict.py
```

### 4. Run Research Version
```bash
python main.py
```

---

## 🎙 Dataset
- RAVDESS Emotional Speech Audio  
- Download: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

---

## 👨‍💻 Author

**Medagam Venkata Karthik Reddy**  
GitHub: [@karthi-reddy-z](https://github.com/karthi-reddy-z)  
Email: karthikreddy136hz@gmail.com

---

## 📜 License
This project is licensed for academic and personal research use.
