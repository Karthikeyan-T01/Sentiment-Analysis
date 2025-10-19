# Sentiment Analysis App 😊

A real-time sentiment analysis application built with TensorFlow and Streamlit.

## Features

- ✨ Real-time sentiment analysis
- 📊 Interactive confidence visualization
- 🎯 82.94% accuracy on test data
- 🚀 Fast predictions with BiLSTM model

## Model Details

- **Architecture**: Bidirectional LSTM
- **Dataset**: Sentiment140 (1.6M tweets)
- **Accuracy**: 82.94%
- **ROC-AUC Score**: 0.9105

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app2.py
```

## Project Structure
```
sentiment-analysis/
├── app2.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── models/
│   ├── sentiment_model.keras  # Trained model
│   ├── tokenizer.pkl          # Tokenizer
│   └── config.pkl             # Configuration
└── README.md
```

## Usage

1. Enter or select sample text
2. Click "Analyze Sentiment"
3. View results with confidence scores

## Technologies Used

- TensorFlow 2.15.0
- Streamlit 1.31.0
- Plotly 5.18.0
- Python 3.9+



