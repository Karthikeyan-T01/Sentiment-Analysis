import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="Sentiment Analyzer Dashboard",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load trained model, tokenizer, and config"""
    try:
        model = tf.keras.models.load_model('./models/sentiment_model.keras')
        
        with open('./models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('./models/config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure model files are in the './models' directory")
        return None, None, None

def preprocess_text(text, tokenizer, max_seq_length):
    """Preprocess text for prediction"""
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_seq_length, 
                          padding='post', truncating='post')
    return padded

def predict_sentiment(text, model, tokenizer, config):
    """Predict sentiment for given text"""
    if not text.strip():
        return None
    
    X = preprocess_text(text, tokenizer, config['max_seq_length'])
    prediction = model.predict(X, verbose=0)[0][0]
    
    sentiment = "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
    confidence = prediction if sentiment == "POSITIVE" else (1 - prediction)
    
    return {
        "sentiment": sentiment,
        "confidence": float(confidence),
        "probability": float(prediction),
        "emoji": "😊" if sentiment == "POSITIVE" else "😞"
    }

def create_gauge_chart(confidence, sentiment):
    """Create gauge chart for confidence visualization"""
    color = "#28a745" if sentiment == "POSITIVE" else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 30}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_distribution_chart(sentiments):
    """Create sentiment distribution chart"""
    if not sentiments:
        return None
    
    sentiment_counts = pd.Series(sentiments).value_counts()
    colors = ['#28a745' if s == 'POSITIVE' else '#dc3545' for s in sentiment_counts.index]
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Count'},
        title='Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545'}
    )
    
    fig.update_layout(showlegend=False, height=400)
    return fig

def main():
    # Header
    st.title("😊 Sentiment Analysis Dashboard")
    st.markdown("Analyze sentiment in social media posts and text data")
    st.markdown("---")
    
    # Load model
    model, tokenizer, config = load_model_and_tokenizer()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"""
        **Model Version:** {config.get('model_version', 'N/A')}
        **Dataset:** {config.get('dataset', 'N/A')}
        **Accuracy:** {config.get('test_accuracy', 0)*100:.2f}%
        **ROC-AUC:** {config.get('roc_auc_score', 0):.4f}
        """)
        
        st.markdown("---")
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. Enter text in the input box
        2. Click 'Analyze Sentiment'
        3. View results and visualization
        
        **Tips:**
        - Works best with 10+ words
        - Handles multiple sentences
        - Real-time analysis
        """)
        
        st.markdown("---")
        st.header("📝 Sample Texts")
        samples = {
            "Positive 1": "I absolutely love this product! Best purchase ever!",
            "Negative 1": "Terrible service and poor quality. Very disappointed.",
            "Positive 2": "Amazing experience! Highly recommended to everyone.",
            "Negative 2": "Worst experience of my life. Will never come back."
        }
        
        selected_sample = st.selectbox("Choose a sample:", [""] + list(samples.keys()))
    
    # Main content - Two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Text Analysis")
        
        if selected_sample and selected_sample in samples:
            default_text = samples[selected_sample]
        else:
            default_text = ""
        
        user_input = st.text_area(
            "Enter text to analyze:",
            value=default_text,
            height=150,
            placeholder="Type or paste social media posts, reviews, or any text...",
            help="Enter text for sentiment analysis"
        )
        
        col_analyze, col_clear = st.columns([3, 1])
        
        with col_analyze:
            analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)
        with col_clear:
            clear_button = st.button("Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
    
    with col2:
        st.subheader("📊 Quick Stats")
        st.metric("Max Text Length", f"{config['max_seq_length']} tokens")
        st.metric("Vocabulary Size", f"{config['max_vocab_size']:,}")
        st.metric("Model Type", "BiLSTM")
    
    # Analysis Results
    if analyze_button and user_input:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(user_input, model, tokenizer, config)
            
            if result:
                st.markdown("---")
                st.subheader("📈 Analysis Results")
                
                # Results columns
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="text-align: center;">{result['emoji']}</h2>
                        <h3 class="{result['sentiment'].lower()}" style="text-align: center;">
                            {result['sentiment']}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                    st.metric("Probability", f"{result['probability']:.4f}")
                
                with res_col2:
                    fig = create_gauge_chart(result['confidence'], result['sentiment'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("---")
                st.subheader("💡 Interpretation")
                
                if result['confidence'] > 0.9:
                    confidence_level = "Very High"
                    interpretation = "The model is very confident about this prediction."
                elif result['confidence'] > 0.75:
                    confidence_level = "High"
                    interpretation = "The model is quite confident about this prediction."
                elif result['confidence'] > 0.6:
                    confidence_level = "Moderate"
                    interpretation = "The model has moderate confidence in this prediction."
                else:
                    confidence_level = "Low"
                    interpretation = "The model is uncertain. The text might be ambiguous or neutral."
                
                st.info(f"**Confidence Level:** {confidence_level} ({result['confidence']*100:.2f}%) - {interpretation}")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Batch Analysis Section
    st.markdown("---")
    st.subheader("📊 Batch Analysis")
    
    batch_col1, batch_col2 = st.columns([2, 1])
    
    with batch_col1:
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            height=150,
            placeholder="Line 1\nLine 2\nLine 3..."
        )
    
    with batch_col2:
        st.write("")
        st.write("")
        batch_button = st.button("🔄 Analyze Batch", use_container_width=True)
    
    if batch_button and batch_text:
        texts = [t.strip() for t in batch_text.split('\n') if t.strip()]
        
        if texts:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                results_list = []
                sentiments = []
                
                for text in texts:
                    result = predict_sentiment(text, model, tokenizer, config)
                    if result:
                        results_list.append({
                            "Text": text[:50] + "..." if len(text) > 50 else text,
                            "Sentiment": result['sentiment'],
                            "Confidence": f"{result['confidence']*100:.2f}%",
                            "Probability": f"{result['probability']:.4f}"
                        })
                        sentiments.append(result['sentiment'])
                
                # Display results
                st.dataframe(pd.DataFrame(results_list), use_container_width=True)
                
                # Distribution chart
                if sentiments:
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        sentiment_counts = pd.Series(sentiments).value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_chart2:
                        stats = {
                            "Total Texts": len(texts),
                            "Positive": sentiments.count("POSITIVE"),
                            "Negative": sentiments.count("NEGATIVE"),
                            "Positive %": f"{sentiments.count('POSITIVE')/len(texts)*100:.1f}%"
                        }
                        for key, value in stats.items():
                            st.metric(key, value)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with ❤️ using Streamlit & TensorFlow</p>
        <p style="font-size: 12px;">Model trained on Sentiment140 dataset with BiLSTM architecture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
