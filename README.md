# IMDb Sentiment Analyzer

An AI-powered web application that classifies IMDb movie reviews as Positive or Negative using a Bidirectional LSTM (Bi-LSTM) deep learning model.

## 🚀 Live Demo

IMDb Sentiment Analyzer:
https://imdb-sentiment-analyzer-9jeitus8bvly9oqp4jyvvf.streamlit.app/

## 📌 Project Overview

Sentiment Analysis is an important Natural Language Processing (NLP) task used to determine the emotional polarity of text.

In this project, four deep learning models were developed and evaluated:

- RNN
- LSTM
- Bi-LSTM
- GRU

After comparing their performance, Bi-LSTM was selected as the final model because it achieved the best overall predictive performance.

## 🏆 Model Performance

| Metric | Bi-LSTM |
|---|---:|
| Accuracy | 78.91% |
| Precision | 79.32% |
| Recall | 78.21% |
| F1-Score | 78.76% |
| MCC | 0.5783 |
| Cohen's Kappa | 0.5782 |
| ROC-AUC | 0.8657 |
| PR-AUC | 0.8570 |

### Model Comparison

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| RNN | 74.58% | 0.7330 | 0.7998 |
| LSTM | 78.34% | 0.7747 | 0.8488 |
| Bi-LSTM | 78.91% | 0.7876 | 0.8657 |
| GRU | 77.75% | 0.7721 | 0.8446 |

## 🧠 Bi-LSTM Architecture

The deployed model uses:

- Vocabulary Size: 10,000
- Sequence Length: 100
- Embedding Dimension: 128
- Bi-LSTM Units: 64 in each direction
- Output: Dense layer with Sigmoid activation
- Parameters: Approximately 1.38 million

### Prediction

The model generates a probability between 0 and 1.

Probability >= 0.5 → Positive
Probability < 0.5 → Negative

## ✨ Features

- IMDb movie review sentiment analysis
- Bi-LSTM deep learning model
- Positive/Negative classification
- Prediction confidence
- Positive probability
- Word count
- Positive, negative and mixed example reviews
- Interactive Streamlit web interface
- Model performance information

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- scikit-learn
- Pandas
- Regular Expressions
- Pickle

## 📂 Project Structure

IMDb-Sentiment-Analyzer/
│
├── app.py
├── bilstm_sentiment.weights.h5
├── word_index.pkl
├── requirements.txt
├── README.md
└── .gitignore

## ⚙️ Run Locally

### 1. Clone the repository

git clone https://github.com/TPS1234795/IMDb-Sentiment-Analyzer.git
cd IMDb-Sentiment-Analyzer

### 2. Create a virtual environment

python -m venv venv

### 3. Activate the virtual environment

Windows PowerShell:

venv\Scripts\activate

Linux/macOS:

source venv/bin/activate

### 4. Install dependencies

pip install -r requirements.txt

### 5. Run the application

streamlit run app.py

The application will open at:

http://localhost:8501

## 📊 Dataset

The project uses the IMDb 50K movie review dataset:

- 25,000 training reviews
- 25,000 testing reviews
- Binary sentiment classification:
  - Positive
  - Negative

## 🔄 Prediction Pipeline

Movie Review
     ↓
Text Cleaning
     ↓
Word Mapping
     ↓
Sequence Conversion
     ↓
Pre-Padding
     ↓
Embedding Layer
     ↓
Bi-LSTM
     ↓
Sigmoid Output
     ↓
Positive / Negative

## 🌐 Deployment

The application is deployed using Streamlit Community Cloud.

Live Application:

https://imdb-sentiment-analyzer-9jeitus8bvly9oqp4jyvvf.streamlit.app/

## 🎯 Use Cases

This project can be used for:

- Movie review sentiment analysis
- NLP demonstrations
- Deep learning experiments
- Text classification
- RNN/LSTM architecture comparison
- Academic projects
- Machine Learning portfolio demonstrations

## 📈 Why Bi-LSTM?

Among the evaluated models, Bi-LSTM achieved the strongest overall predictive performance.

It obtained the highest:

- Accuracy
- F1-score
- MCC
- ROC-AUC
- PR-AUC
- Recall

Although Bi-LSTM requires more computational resources than RNN, LSTM and GRU, its improved predictive performance makes it the best model for this experiment.

## 🔮 Future Improvements

- Implement BERT/Transformer-based sentiment classification
- Add attention mechanism
- Improve accuracy through hyperparameter tuning
- Add batch prediction
- Add CSV upload for multiple reviews
- Add downloadable prediction reports
- Add sentiment visualization
- Compare with modern Transformer models

## 👩‍💻 Author

TANIPRAVA SAHOO

B.Tech – Computer Science & Engineering (AI & ML)

GitHub:
https://github.com/TPS1234795

## ⭐ Acknowledgement

This project was developed as an academic/deep learning project for studying and comparing recurrent neural network architectures for sentiment classification.

If you find this project useful, please star the repository.
