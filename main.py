{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_x403KwAFOoy",
        "outputId": "109bdf93-3c84-4143-bb0c-f0e8d430cbca"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "\n",
            "[notice] A new release of pip is available: 24.0 -> 25.1.1\n",
            "[notice] To update, run: C:\\Users\\karth\\AppData\\Local\\Microsoft\\WindowsApps\\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\\python.exe -m pip install --upgrade pip\n"
          ]
        }
      ],
      "source": [
        "!pip install librosa==0.9.2 resampy==0.4.2 scikit-learn --quiet\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "EQDavSIsJbuC",
        "outputId": "7b017439-3db7-411d-c92a-ad7237dcc4ea"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n",
            "replace /content/RAVDESS/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav? [y]es, [n]o, [A]ll, [N]one, [r]ename: n\n",
            "replace /content/RAVDESS/RAVDESS/Actor_01/03-01-01-01-01-02-01.wav? [y]es, [n]o, [A]ll, [N]one, [r]ename: n\n",
            "replace /content/RAVDESS/RAVDESS/Actor_01/03-01-01-01-02-01-01.wav? [y]es, [n]o, [A]ll, [N]one, [r]ename: A\n",
            "\n",
            "\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "!unzip -q \"/content/drive/MyDrive/RAVDESS.zip\" -d \"/content/RAVDESS\"\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IEA0DZoQJkFa",
        "outputId": "f0edf436-0826-423d-98da-8b798e6695ae"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "✅ MFCC Extraction Complete — Success: 1440, Failed: 0\n",
            "🎉 Preprocessing complete!\n",
            "✅ X_train shape: (1152, 40, 300)\n",
            "✅ y_train shape: (1152,)\n"
          ]
        }
      ],
      "source": [
        "import os\n",
        "import librosa\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "\n",
        "# ✅ Correct path (based on your folder structure)\n",
        "DATASET_PATH = \"/content/RAVDESS/RAVDESS/\"\n",
        "\n",
        "# Emotion label mapping\n",
        "emotion_map = {\n",
        "    '01': 'neutral',\n",
        "    '02': 'calm',\n",
        "    '03': 'happy',\n",
        "    '04': 'sad',\n",
        "    '05': 'angry',\n",
        "    '06': 'fearful',\n",
        "    '07': 'disgust',\n",
        "    '08': 'surprised'\n",
        "}\n",
        "\n",
        "# 🎧 MFCC feature extraction\n",
        "def extract_features(file_path, max_pad_len=300):\n",
        "    try:\n",
        "        audio, sr = librosa.load(file_path, res_type='kaiser_fast')\n",
        "        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)\n",
        "        pad_width = max_pad_len - mfccs.shape[1]\n",
        "        if pad_width > 0:\n",
        "            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')\n",
        "        else:\n",
        "            mfccs = mfccs[:, :max_pad_len]\n",
        "        return mfccs\n",
        "    except Exception as e:\n",
        "        print(f\"❌ Error processing {file_path}: {e}\")\n",
        "        return None\n",
        "\n",
        "# 🔁 Load data\n",
        "X, y = [], []\n",
        "success, fail = 0, 0\n",
        "\n",
        "for actor_folder in sorted(os.listdir(DATASET_PATH)):\n",
        "    actor_path = os.path.join(DATASET_PATH, actor_folder)\n",
        "    if not os.path.isdir(actor_path):\n",
        "        continue\n",
        "    for file in sorted(os.listdir(actor_path)):\n",
        "        if file.endswith(\".wav\"):\n",
        "            file_path = os.path.join(actor_path, file)\n",
        "            emotion_code = file.split(\"-\")[2]\n",
        "            label = emotion_map.get(emotion_code)\n",
        "            if label:\n",
        "                mfcc = extract_features(file_path)\n",
        "                if mfcc is not None:\n",
        "                    X.append(mfcc)\n",
        "                    y.append(label)\n",
        "                    success += 1\n",
        "                else:\n",
        "                    fail += 1\n",
        "\n",
        "print(f\"✅ MFCC Extraction Complete — Success: {success}, Failed: {fail}\")\n",
        "\n",
        "# 🧠 Encode labels\n",
        "X = np.array(X)\n",
        "y = np.array(y)\n",
        "\n",
        "le = LabelEncoder()\n",
        "y_encoded = le.fit_transform(y)\n",
        "\n",
        "# 📊 Train-test split\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42\n",
        ")\n",
        "\n",
        "# 💾 Save data\n",
        "np.save('X_train.npy', X_train)\n",
        "np.save('X_test.npy', X_test)\n",
        "np.save('y_train.npy', y_train)\n",
        "np.save('y_test.npy', y_test)\n",
        "np.save('label_classes.npy', le.classes_)\n",
        "\n",
        "print(\"🎉 Preprocessing complete!\")\n",
        "print(\"✅ X_train shape:\", X_train.shape)\n",
        "print(\"✅ y_train shape:\", y_train.shape)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1oJI2bDlLKi2",
        "outputId": "771b67f9-0cbe-495c-b838-91eddde8a39e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "✅ Data shapes:\n",
            "X_train: (1152, 40, 300)\n",
            "y_train: (1152,)\n",
            "Classes: ['angry' 'calm' 'disgust' 'fearful' 'happy' 'neutral' 'sad' 'surprised']\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "\n",
        "# Load processed data\n",
        "X_train = np.load('X_train.npy')\n",
        "X_test = np.load('X_test.npy')\n",
        "y_train = np.load('y_train.npy')\n",
        "y_test = np.load('y_test.npy')\n",
        "label_classes = np.load('label_classes.npy')\n",
        "\n",
        "print(\"✅ Data shapes:\")\n",
        "print(\"X_train:\", X_train.shape)\n",
        "print(\"y_train:\", y_train.shape)\n",
        "print(\"Classes:\", label_classes)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oymEJwKsLVqP",
        "outputId": "b3e0dc1d-5cfd-4c9e-d550-ec0099beac71"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m363.4/363.4 MB\u001b[0m \u001b[31m1.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m13.8/13.8 MB\u001b[0m \u001b[31m108.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m24.6/24.6 MB\u001b[0m \u001b[31m91.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m883.7/883.7 kB\u001b[0m \u001b[31m58.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m664.8/664.8 MB\u001b[0m \u001b[31m1.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m211.5/211.5 MB\u001b[0m \u001b[31m5.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m56.3/56.3 MB\u001b[0m \u001b[31m14.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m127.9/127.9 MB\u001b[0m \u001b[31m7.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.5/207.5 MB\u001b[0m \u001b[31m5.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m21.1/21.1 MB\u001b[0m \u001b[31m103.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "cuml-cu12 25.2.1 requires numba<0.61.0a0,>=0.59.1, but you have numba 0.61.2 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0m"
          ]
        }
      ],
      "source": [
        "!pip install torch torchvision torchaudio --quiet\n",
        "!pip install torchdiffeq --quiet\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "4QkTguB6LWlI"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "from torchdiffeq import odeint\n",
        "\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "\n",
        "# 🔁 Spike function (SNN-like behavior)\n",
        "class SpikeFunction(torch.autograd.Function):\n",
        "    @staticmethod\n",
        "    def forward(ctx, input):\n",
        "        ctx.save_for_backward(input)\n",
        "        return (input > 0).float()\n",
        "\n",
        "    @staticmethod\n",
        "    def backward(ctx, grad_output):\n",
        "        input, = ctx.saved_tensors\n",
        "        grad_input = grad_output.clone()\n",
        "        grad_input[input.abs() > 1] = 0\n",
        "        return grad_input\n",
        "\n",
        "spike_fn = SpikeFunction.apply\n",
        "\n",
        "# 🧠 ODE block\n",
        "class ODEFunc(nn.Module):\n",
        "    def __init__(self, dim):\n",
        "        super().__init__()\n",
        "        self.linear = nn.Linear(dim, dim)\n",
        "\n",
        "    def forward(self, t, x):\n",
        "        return self.linear(x)\n",
        "\n",
        "# 🎯 Full EmotionNet\n",
        "class EmotionNet(nn.Module):\n",
        "    def __init__(self, num_classes):\n",
        "        super(EmotionNet, self).__init__()\n",
        "        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2)\n",
        "        self.pool = nn.MaxPool2d((2, 2))\n",
        "        self.dropout = nn.Dropout(0.3)\n",
        "\n",
        "        self.gru = nn.GRU(input_size=3000, hidden_size=128, batch_first=True)\n",
        "\n",
        "        self.odefunc = ODEFunc(128)\n",
        "        self.fc = nn.Linear(128, num_classes)\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = x.unsqueeze(1)  # (B, 1, 40, 300)\n",
        "        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 20, 150)\n",
        "        x = self.dropout(x)\n",
        "\n",
        "        x = spike_fn(x)  # Apply SNN-like spiking\n",
        "        x = x.view(x.size(0), 16, -1)  # Flatten conv output to (B, 16, features)\n",
        "        x, _ = self.gru(x)  # Pass through GRU\n",
        "        x = x[:, -1, :]     # Take last time step output\n",
        "\n",
        "        x = odeint(self.odefunc, x, torch.tensor([0, 1]).float().to(x.device))[-1]\n",
        "        x = self.fc(x)\n",
        "        return x\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qUexFjbALY45",
        "outputId": "eb926d42-8136-4f4a-e7d5-1929395587cc"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "✅ Data ready for training!\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "from torch.utils.data import DataLoader, TensorDataset\n",
        "\n",
        "# 🔁 Convert numpy to PyTorch tensors\n",
        "X_train_tensor = torch.tensor(X_train, dtype=torch.float32)\n",
        "X_test_tensor = torch.tensor(X_test, dtype=torch.float32)\n",
        "y_train_tensor = torch.tensor(y_train, dtype=torch.long)\n",
        "y_test_tensor = torch.tensor(y_test, dtype=torch.long)\n",
        "\n",
        "# 📦 Create Datasets\n",
        "train_dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
        "test_dataset = TensorDataset(X_test_tensor, y_test_tensor)\n",
        "\n",
        "# 🚀 Create DataLoaders\n",
        "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
        "test_loader = DataLoader(test_dataset, batch_size=32)\n",
        "\n",
        "print(\"✅ Data ready for training!\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Nh1FiK3SMtKN",
        "outputId": "a8e46b4a-a2ed-4dc7-cbec-9010bccdee58"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "📅 Epoch 1/15 — Loss: 2.0730 | Train Acc: 0.1840 | Test Acc: 0.1910\n",
            "📅 Epoch 2/15 — Loss: 1.9095 | Train Acc: 0.2543 | Test Acc: 0.2326\n",
            "📅 Epoch 3/15 — Loss: 1.8177 | Train Acc: 0.3151 | Test Acc: 0.2326\n",
            "📅 Epoch 4/15 — Loss: 1.6900 | Train Acc: 0.3689 | Test Acc: 0.3194\n",
            "📅 Epoch 5/15 — Loss: 1.5637 | Train Acc: 0.3984 | Test Acc: 0.3333\n",
            "📅 Epoch 6/15 — Loss: 1.4882 | Train Acc: 0.4245 | Test Acc: 0.3056\n",
            "📅 Epoch 7/15 — Loss: 1.3591 | Train Acc: 0.4688 | Test Acc: 0.3646\n",
            "📅 Epoch 8/15 — Loss: 1.3691 | Train Acc: 0.4913 | Test Acc: 0.4097\n",
            "📅 Epoch 9/15 — Loss: 1.1662 | Train Acc: 0.5616 | Test Acc: 0.4097\n",
            "📅 Epoch 10/15 — Loss: 1.0802 | Train Acc: 0.6042 | Test Acc: 0.4132\n",
            "📅 Epoch 11/15 — Loss: 0.9505 | Train Acc: 0.6658 | Test Acc: 0.4201\n",
            "📅 Epoch 12/15 — Loss: 0.8459 | Train Acc: 0.6988 | Test Acc: 0.4236\n",
            "📅 Epoch 13/15 — Loss: 0.7205 | Train Acc: 0.7509 | Test Acc: 0.4583\n",
            "📅 Epoch 14/15 — Loss: 0.6424 | Train Acc: 0.7622 | Test Acc: 0.4236\n",
            "📅 Epoch 15/15 — Loss: 0.6116 | Train Acc: 0.7795 | Test Acc: 0.4410\n"
          ]
        }
      ],
      "source": [
        "# 🔥 Load model\n",
        "num_classes = len(label_classes)\n",
        "model = EmotionNet(num_classes=num_classes).to(device)\n",
        "\n",
        "# ⚙️ Define optimizer and loss\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n",
        "\n",
        "# 📈 Training loop\n",
        "def train(model, loader):\n",
        "    model.train()\n",
        "    total_loss, correct = 0, 0\n",
        "    for x, y in loader:\n",
        "        x, y = x.to(device), y.to(device)\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(x)\n",
        "        loss = criterion(outputs, y)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        total_loss += loss.item()\n",
        "        correct += (outputs.argmax(1) == y).sum().item()\n",
        "    return total_loss / len(loader), correct / len(loader.dataset)\n",
        "\n",
        "# 🔍 Evaluation\n",
        "def evaluate(model, loader):\n",
        "    model.eval()\n",
        "    correct = 0\n",
        "    with torch.no_grad():\n",
        "        for x, y in loader:\n",
        "            x, y = x.to(device), y.to(device)\n",
        "            outputs = model(x)\n",
        "            correct += (outputs.argmax(1) == y).sum().item()\n",
        "    return correct / len(loader.dataset)\n",
        "\n",
        "# 🔁 Run training\n",
        "EPOCHS = 15\n",
        "for epoch in range(EPOCHS):\n",
        "    train_loss, train_acc = train(model, train_loader)\n",
        "    test_acc = evaluate(model, test_loader)\n",
        "    print(f\"📅 Epoch {epoch+1}/{EPOCHS} — Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "Ci3RTpkjOAWE",
        "outputId": "d948ce9e-5454-46c1-ac9b-7e9da09df2fe"
      },
      "outputs": [
        {
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAwwAAAK9CAYAAACJnusfAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAp+RJREFUeJzs3Xd8U+XfxvErLdAWSgdllDLKlF323kOWoCwHONggoihLQdkyFGWrLGWDioKoiMgWVEQZgiIgowIChVIoUFpKafP8wUN+jSXSYuE+sZ+3r7wkd05yrp4mab65x7HZ7Xa7AAAAAOA2PEwHAAAAAGBdFAwAAAAAXKJgAAAAAOASBQMAAAAAlygYAAAAALhEwQAAAADAJQoGAAAAAC5RMAAAAABwiYIBAAAAgEsUDICFHD58WE2bNpW/v79sNptWrVqVro//559/ymazacGCBen6uO6sQYMGatCggekY/ymFChVSly5dTMfIMGw2m0aNGpWqbfndALgbFAzA3xw9elS9e/dWkSJF5O3tLT8/P9WuXVvTpk1TXFzcPd13586d9euvv2rcuHFavHixqlSpck/3dz916dJFNptNfn5+tz2Ohw8fls1mk81m09tvv53mxz99+rRGjRqlX375JR3SWsutQs/V5Y033rjvmX744QeNGjVK0dHR933frixYsMDpuGTKlEn58uVTly5ddOrUKdPx7hsr/m4AuLdMpgMAVvLVV1/p0UcflZeXl5555hmVLVtW169f13fffafBgwdr//79mjNnzj3Zd1xcnLZv367XXntNzz///D3ZR2hoqOLi4pQ5c+Z78vh3kilTJsXGxurLL7/UY4895nTb0qVL5e3trWvXrt3VY58+fVqjR49WoUKFVKFChVTfb926dXe1PxM6duyoli1bpmivWLHifc/yww8/aPTo0erSpYsCAgKcbjt06JA8PMx9HzVmzBgVLlxY165d048//qgFCxbou+++02+//SZvb29jue6VuLg4Zcr0vz/nVv7dAHBPFAzA/wsPD9cTTzyh0NBQbdq0SXnz5nXc1rdvXx05ckRfffXVPdt/ZGSkJKX4A5+ebDab0Q9MXl5eql27tj788MMUBcOyZcv00EMPacWKFfclS2xsrLJmzaosWbLcl/2lh0qVKumpp54yHeOOvLy8jO6/RYsWjt65Hj16KGfOnHrzzTf1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2mnhwoU6duzYP550avz48Ro+fLi6deum119/XTly5JCHh4deeumlVPekSDePT1rs2bNH586dkyT9+uuvTt8eu1K1atUUxWJqhYeH33Zy+91w9TtKj9/d/fBvc1arVs2xSlKbNm1Up04dderUSYcOHZKvr6/jeTNo0CA1a9bsto/h6kP8/ZRer9n05C7PIQDpi4IB+H+tWrXSnDlztH37dqfhHLcTGhqqpKQkHT582PFNnHRzAmF0dLRjxaP0EBgYeNsTMN3ug6mHh4caN26sxo0ba/LkyRo/frxee+01bd68+bbfTN/KeejQoRS3HTx4UDlz5kzzt8mp1alTJ82bN08eHh564oknXG736aefqmHDhvrggw+c2qOjox0TOqXUF2+pcfXqVXXt2lWlS5dWrVq1NHHiRLVt29axEpMrS5cuveuT+wUHB9/V/dJTWp4PaTneoaGhLh8z+X7vBU9PT02YMEENGzbUO++8oyFDhqhIkSKSbg4bulOPTdGiRe/4RUJoaKj27dunpKQkp16Gv/98oaGh2rx5s2NJ31vSe5Uhd/ndAHAfzGEA/t/LL7+sbNmyqUePHjp79myK248ePapp06ZJkuPkWVOnTnXaZvLkyZKkhx56KN1yFS1aVJcuXdK+ffscbWfOnEmxEtOFCxdS3PfWCcz+vtTrLXnz5lWFChW0cOFCp6Lkt99+07p16257krD00rBhQ73++ut65513/vHDsqenZ4pvLz/55JMUZ+699UE2Pc5u+8orr+jEiRNauHChJk+erEKFCqlz584uj+MttWvXVpMmTe7qYoUTiqXl+ZCW492yZUv99NNP2r59u6Pt6tWrmjNnjgoVKpRiDkZ6a9CggapVq6apU6fq2rVryp07txo0aKDZs2frzJkzKba/dU4USWrfvr327t2b4vUm/e9b9ZYtWyoiIkIff/yx47YbN25oxowZ8vX1Vf369SXdXJo3ISFBc+fOdWyXlJSkd999N91+Vsm9fjcA3AM9DMD/K1q0qJYtW6bHH39cpUqVcjrT8w8//OBYJlGSypcvr86dO2vOnDmKjo5W/fr19dNPP2nhwoVq06aNGjZsmG65nnjiCb3yyitq27at+vXrp9jYWM2cOVMPPPCA06TfMWPGaOvWrXrooYcUGhqqc+fO6b333lP+/PmdJjT+3VtvvaUWLVqoZs2a6t69u2MZ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",
            "text/plain": [
              "<Figure size 1000x800 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "📊 Classification Report:\n",
            "\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "       angry       0.75      0.55      0.64        38\n",
            "        calm       0.52      0.63      0.57        38\n",
            "     disgust       0.47      0.24      0.32        38\n",
            "     fearful       0.45      0.64      0.53        39\n",
            "       happy       0.28      0.26      0.27        39\n",
            "     neutral       0.22      0.58      0.32        19\n",
            "         sad       0.33      0.16      0.21        38\n",
            "   surprised       0.58      0.54      0.56        39\n",
            "\n",
            "    accuracy                           0.44       288\n",
            "   macro avg       0.45      0.45      0.43       288\n",
            "weighted avg       0.47      0.44      0.43       288\n",
            "\n"
          ]
        }
      ],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.metrics import confusion_matrix, classification_report\n",
        "\n",
        "# 📌 Get all predictions\n",
        "all_preds = []\n",
        "all_labels = []\n",
        "\n",
        "model.eval()\n",
        "with torch.no_grad():\n",
        "    for x, y in test_loader:\n",
        "        x = x.to(device)\n",
        "        outputs = model(x)\n",
        "        preds = outputs.argmax(1).cpu().numpy()\n",
        "        all_preds.extend(preds)\n",
        "        all_labels.extend(y.numpy())\n",
        "\n",
        "# 🧮 Confusion matrix\n",
        "cm = confusion_matrix(all_labels, all_preds)\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_classes, yticklabels=label_classes)\n",
        "plt.xlabel(\"Predicted Label\")\n",
        "plt.ylabel(\"True Label\")\n",
        "plt.title(\"Confusion Matrix — Emotion Recognition\")\n",
        "plt.show()\n",
        "\n",
        "# 📋 Classification report\n",
        "print(\"\\n📊 Classification Report:\\n\")\n",
        "print(classification_report(all_labels, all_preds, target_names=label_classes))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LEJOBzmyPDBs",
        "outputId": "6900e4de-7b12-4970-e964-d2119d11e138"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Reading package lists... Done\n",
            "Building dependency tree... Done\n",
            "Reading state information... Done\n",
            "ffmpeg is already the newest version (7:4.4.2-0ubuntu0.22.04.1).\n",
            "0 upgraded, 0 newly installed, 0 to remove and 35 not upgraded.\n"
          ]
        }
      ],
      "source": [
        "# 📦 Step 1: Install ffmpeg to handle audio conversion\n",
        "!apt install ffmpeg -y\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 20,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VmbnY9WaP4nB",
        "outputId": "da4f6cac-2e9a-4c09-8ffa-87c780aaaa26"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c) 2000-2021 the FFmpeg developers\n",
            "  built with gcc 11 (Ubuntu 11.2.0-19ubuntu1)\n",
            "  configuration: --prefix=/usr --extra-version=0ubuntu0.22.04.1 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-libpulse --enable-librabbitmq --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libsrt --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx --enable-openal --enable-opencl --enable-opengl --enable-sdl2 --enable-pocketsphinx --enable-librsvg --enable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-libx264 --enable-shared\n",
            "  libavutil      56. 70.100 / 56. 70.100\n",
            "  libavcodec     58.134.100 / 58.134.100\n",
            "  libavformat    58. 76.100 / 58. 76.100\n",
            "  libavdevice    58. 13.100 / 58. 13.100\n",
            "  libavfilter     7.110.100 /  7.110.100\n",
            "  libswscale      5.  9.100 /  5.  9.100\n",
            "  libswresample   3.  9.100 /  3.  9.100\n",
            "  libpostproc    55.  9.100 / 55.  9.100\n",
            "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from '/content/TEST.unknown':\n",
            "  Metadata:\n",
            "    major_brand     : mp42\n",
            "    minor_version   : 0\n",
            "    compatible_brands: isommp42\n",
            "    creation_time   : 2025-07-08T13:03:52.000000Z\n",
            "    com.android.version: 13\n",
            "  Duration: 00:00:05.57, start: 0.000000, bitrate: 100 kb/s\n",
            "  Stream #0:0(eng): Audio: aac (LC) (mp4a / 0x6134706D), 16000 Hz, mono, fltp, 96 kb/s (default)\n",
            "    Metadata:\n",
            "      creation_time   : 2025-07-08T13:03:52.000000Z\n",
            "      handler_name    : SoundHandle\n",
            "      vendor_id       : [0][0][0][0]\n",
            "Stream mapping:\n",
            "  Stream #0:0 -> #0:0 (aac (native) -> pcm_s16le (native))\n",
            "Press [q] to stop, [?] for help\n",
            "Output #0, wav, to '/content/TEST.unknown.wav':\n",
            "  Metadata:\n",
            "    major_brand     : mp42\n",
            "    minor_version   : 0\n",
            "    compatible_brands: isommp42\n",
            "    com.android.version: 13\n",
            "    ISFT            : Lavf58.76.100\n",
            "  Stream #0:0(eng): Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s (default)\n",
            "    Metadata:\n",
            "      creation_time   : 2025-07-08T13:03:52.000000Z\n",
            "      handler_name    : SoundHandle\n",
            "      vendor_id       : [0][0][0][0]\n",
            "      encoder         : Lavc58.134.100 pcm_s16le\n",
            "size=       2kB time=00:00:00.00 bitrate=N/A speed=   0x    \rsize=     174kB time=00:00:05.50 bitrate= 259.1kbits/s speed=1.24e+03x    \n",
            "video:0kB audio:174kB subtitle:0kB other streams:0kB global headers:0kB muxing overhead: 0.043777%\n"
          ]
        }
      ],
      "source": [
        "# 🔁 Step 2: Convert your MP4 (voice recording) to WAV\n",
        "!ffmpeg -i /content/TEST.unknown /content/TEST.unknown.wav\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "aGprjVDEQR_G"
      },
      "outputs": [],
      "source": [
        "# 📚 Step 3: Import required libraries\n",
        "from torchdiffeq import odeint\n",
        "import librosa\n",
        "import numpy as np\n",
        "import torch\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "QJ3P9_vuQUiN"
      },
      "outputs": [],
      "source": [
        "# 🎯 Step 4: Predict Emotion Function\n",
        "def predict_emotion(file_path, model, label_classes):\n",
        "    y, sr = librosa.load(file_path, sr=22050)\n",
        "    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)\n",
        "\n",
        "    # Pad or trim to (40, 300)\n",
        "    if mfcc.shape[1] < 300:\n",
        "        pad_width = 300 - mfcc.shape[1]\n",
        "        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')\n",
        "    else:\n",
        "        mfcc = mfcc[:, :300]\n",
        "\n",
        "    # Convert to tensor (1, 1, 40, 300)\n",
        "    input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)\n",
        "\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        x = model.pool(torch.relu(model.conv1(input_tensor)))\n",
        "        x = model.dropout(x)\n",
        "        x = SpikeFunction.apply(x)\n",
        "        x = x.view(x.size(0), 16, -1)\n",
        "        x, _ = model.gru(x)\n",
        "        x = x[:, -1, :]\n",
        "        x = odeint(model.odefunc, x, torch.tensor([0, 1]).float().to(device))[-1]\n",
        "        output = model.fc(x)\n",
        "\n",
        "    pred = output.argmax(1).item()\n",
        "    return label_classes[pred]\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QxTUnSCvQWlA",
        "outputId": "f2d27323-b607-4819-f258-8a5b28ca12ab"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "🎧 Predicted Emotion: disgust\n"
          ]
        }
      ],
      "source": [
        "# 🧠 Step 5: Run prediction\n",
        "predicted_emotion = predict_emotion(\"/content/TEST.unknown.wav\", model, label_classes)\n",
        "print(f\"🎧 Predicted Emotion: {predicted_emotion}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 27,
      "metadata": {
        "id": "P2VfGpW0Qq3u"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "\n",
        "# 🎯 RAVDESS path\n",
        "DATASET_DIR = \"/content/RAVDESS/RAVDESS\"\n",
        "\n",
        "# ✅ To store only one file per emotion\n",
        "seen_emotions = set()\n",
        "test_files = []\n",
        "\n",
        "# 🔁 Go through dataset and pick 1 sample for each emotion\n",
        "for actor in sorted(os.listdir(DATASET_DIR)):\n",
        "    actor_path = os.path.join(DATASET_DIR, actor)\n",
        "    for file in sorted(os.listdir(actor_path)):\n",
        "        if file.endswith(\".wav\"):\n",
        "            emotion_code = file.split(\"-\")[2]\n",
        "            if emotion_code not in seen_emotions:\n",
        "                file_path = os.path.join(actor_path, file)\n",
        "                test_files.append((file, file_path, emotion_map[emotion_code]))\n",
        "                seen_emotions.add(emotion_code)\n",
        "            if len(seen_emotions) == 8:\n",
        "                break\n",
        "    if len(seen_emotions) == 8:\n",
        "        break\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "3wZyELDZROAz",
        "outputId": "e7b140be-c121-4193-ec27-88057d1b0217"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "🎧 Testing one sample per emotion:\n",
            "\n",
            "✅ 03-01-01-01-01-01-01.wav | 🎯 True: neutral    | 🤖 Predicted: neutral\n",
            "✅ 03-01-02-01-01-01-01.wav | 🎯 True: calm       | 🤖 Predicted: calm\n",
            "✅ 03-01-03-01-01-01-01.wav | 🎯 True: happy      | 🤖 Predicted: happy\n",
            "✅ 03-01-04-01-01-01-01.wav | 🎯 True: sad        | 🤖 Predicted: sad\n",
            "✅ 03-01-05-01-01-01-01.wav | 🎯 True: angry      | 🤖 Predicted: angry\n",
            "✅ 03-01-06-01-01-01-01.wav | 🎯 True: fearful    | 🤖 Predicted: fearful\n",
            "✅ 03-01-07-01-01-01-01.wav | 🎯 True: disgust    | 🤖 Predicted: disgust\n",
            "✅ 03-01-08-01-01-01-01.wav | 🎯 True: surprised  | 🤖 Predicted: surprised\n"
          ]
        }
      ],
      "source": [
        "# 🤖 Predict and display results\n",
        "print(\"🎧 Testing one sample per emotion:\\n\")\n",
        "for fname, path, true_emotion in test_files:\n",
        "    pred_emotion = predict_emotion(path, model, label_classes)\n",
        "    status = \"✅\" if pred_emotion == true_emotion else \"❌\"\n",
        "    print(f\"{status} {fname} | 🎯 True: {true_emotion:10s} | 🤖 Predicted: {pred_emotion}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5HpzOiDiRc8N",
        "outputId": "933d25ca-08fe-4d55-e040-3a2e14454692"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "📊 Accuracy on 1-sample-per-emotion test: 100.00%\n"
          ]
        }
      ],
      "source": [
        "# ✅ Count correct predictions\n",
        "correct = sum(1 for _, _, true in test_files if predict_emotion(_, model, label_classes) == true)\n",
        "\n",
        "# 🎯 Compute accuracy\n",
        "accuracy = correct / 8\n",
        "print(f\"\\n📊 Accuracy on 1-sample-per-emotion test: {accuracy:.2%}\")\n"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.9"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
