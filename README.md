# ML-DL-NLP

# 🤖 AI, ML, DL, NLP – Complete Beginner to Expert Guide (Hinglish)

## 🔰 Table of Contents
1. What is AI/ML/DL/NLP?
2. Difference Between AI, ML, DL, NLP
3. ML Types and Algorithms
4. Deep Learning (Neural Networks)
5. Natural Language Processing (NLP)
6. Real-World Examples & Projects
7. Python Code Examples for Each
8. Full Pipeline for Model Building
9. Model Evaluation Metrics
10. Model Deployment (Flask + Streamlit)
11. Bonus Tools & Resources

---

## 📌 1. What is AI, ML, DL, NLP?

### 🤖 Artificial Intelligence (AI):
Machines ko smart decision lene layak banana. Example: Siri, Alexa

### 🧠 Machine Learning (ML):
Machines khud seekhein data se bina explicitly program kiye.

### 🔥 Deep Learning (DL):
Neural Networks use karke ML ka advance form. Example: Self-driving cars

### 🗣️ Natural Language Processing (NLP):
Human language ko samajhne aur generate karne wali AI. Example: ChatGPT

---

## 🔍 2. AI vs ML vs DL vs NLP: Differences

| Feature             | AI                      | ML                      | DL                          | NLP                      |
|---------------------|--------------------------|--------------------------|------------------------------|--------------------------|
| Level               | Broadest                 | Subset of AI             | Subset of ML                 | Part of AI (Language)    |
| Data Dependency     | Medium                   | High                     | Very High                    | High                     |
| Example             | Game Bot, ChatBot        | Email Spam Filter        | Face Recognition             | Translation, Chatbots    |
| Key Tools           | All-inclusive            | Scikit-learn, pandas     | TensorFlow, PyTorch          | NLTK, spaCy, Transformers|

---

## 🔠 3. Types of Machine Learning + Examples

### 📘 Supervised Learning
- Example: House Price Prediction
- Algorithms:
  - Linear/Logistic Regression
  - Decision Trees
  - Random Forest
  - SVM
  - KNN

### 📙 Unsupervised Learning
- Example: Customer Segmentation
- Algorithms:
  - KMeans Clustering
  - PCA (Dimensionality Reduction)

### 📕 Reinforcement Learning
- Example: Game Playing Bots
- Algorithms:
  - Q-Learning
  - Deep Q Networks

---

## 💡 4. Deep Learning (DL)

### 🧱 Basic Concepts:
- Neuron
- Layer (Input, Hidden, Output)
- Activation Functions (ReLU, Sigmoid)
- Loss Functions (MSE, CrossEntropy)

### 🛠️ Frameworks:
- TensorFlow
- PyTorch

### 🔁 Neural Networks:
- ANN (Simple neural network)
- CNN (Image Processing)
- RNN (Sequence/Text)
- LSTM/GRU (Long-term memory for sequences)

---

## 🧠 5. Natural Language Processing (NLP)

### 📗 Core Concepts:
- Tokenization
- Lemmatization/Stemming
- Bag of Words / TF-IDF
- Word2Vec / BERT / GPT

### 🧪 NLP Tasks:
- Sentiment Analysis
- Text Classification
- Named Entity Recognition (NER)
- Translation
- Text Generation (ChatGPT)

---

## 🧪 6. Real Projects (Mini + Major)

### 🏠 House Price Prediction (Regression)
### 📧 Spam Detection (Classification)
### 🎥 Movie Recommendation (Collaborative Filtering)
### 👨‍⚕️ Disease Prediction (Classification)
### 🤖 Chatbot using NLP (NLP + DL)

---

## 🧰 7. Code Examples

### ✅ ML (Classification)
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### ✅ DL (Neural Network - TensorFlow)
```python
import tensorflow as tf
model = tf.keras.Sequential([
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X_train, y_train, epochs=10)
```

### ✅ NLP (Text Classification)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)
```

---

## 🏗️ 8. Full Pipeline (ML/DL)

1. Problem Statement
2. Data Collection (CSV, API, SQL)
3. EDA (Nulls, Outliers)
4. Data Cleaning
5. Feature Engineering
6. Scaling/Encoding
7. Model Selection
8. Train-Test Split
9. Training + Evaluation
10. Cross-validation
11. Model Saving
12. Deployment (Flask/Streamlit)

---

## 🎯 9. Evaluation Metrics

### 📌 Classification:
- Accuracy, Precision, Recall, F1
- Confusion Matrix

### 📌 Regression:
- MAE, MSE, RMSE
- R² Score

### 📌 DL:
- Loss vs Epochs Graph
- Validation Accuracy

---

## 🚀 10. Deployment Examples

### ✅ Flask Deployment
```python
from flask import Flask, request, jsonify
import joblib
model = joblib.load('model.pkl')

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pred = model.predict([data['features']])
    return jsonify({'prediction': int(pred[0])})
```

### ✅ Streamlit Deployment
```python
import streamlit as st
model = joblib.load('model.pkl')
input = st.text_input("Enter text")
pred = model.predict([input])
st.write(f"Prediction: {pred[0]}")
```

---

## 🎁 11. Tools & Resources

- [scikit-learn](https://scikit-learn.org)
- [Keras](https://keras.io)
- [Hugging Face (Transformers)](https://huggingface.co/models)
- [Kaggle](https://www.kaggle.com)
- [Google Colab](https://colab.research.google.com)

---

