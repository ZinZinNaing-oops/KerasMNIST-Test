# KERAS-MNIST-TEST

This repository contains a simple Keras/TensorFlow example for training and testing a **CNN on the MNIST handwritten digits dataset**.

Files:

- `mnist_cnn_keras.py` – trains a CNN on MNIST and saves the model
- `mnist_cnn_model.h5` – saved trained model (created by the training script)
- `test_saved_model.py` – loads the saved model and runs evaluation / prediction
- `requirements.txt` – Python dependencies

---

## Requirements

- Python **3.11** (recommended, TensorFlow 2.16 compatible)
- pip
- (Optional but recommended) virtual environment (`venv`)

> On Apple Silicon (M1–M4, including), `requirements.txt` is already set up to use  
> `tensorflow-macos` + `tensorflow-metal`.  
> On Windows / Linux / Intel Mac, it will install normal `tensorflow`.

---

## Environment Setup / 環境構築

### 1. Clone the repository

```bash
git clone https://github.com/ZinZinNaing-oops/KerasMNIST-Test.git
cd KerasMNIST-Test

## ▶️ How to Run the Project (Step by Step)

### 1️⃣ Train the CNN model

Run the training script:

```bash
python mnist_cnn_keras.py
