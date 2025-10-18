import streamlit as st
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import plotly.graph_objects as go
import time

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
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
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .sentiment-positive {
        color: #28a745;
        font-size: 24px;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-size: 24px;
        font-weight: bold;
    }
    .confidence-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model, tokenizer, and config"""
    try:
        model = tf.keras.models.load_model('./models/sentiment_model.keras')
        
        with open('./models/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open('./models/config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        return model, tokenizer, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Prediction function
def predict_sentiment(text, model, tokenizer, max_seq_length):
    """Predict sentiment for a given text"""
    if not text.strip():
        return None
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_seq_length, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    # Determine sentiment
    sentiment = "POSITIVE" if prediction >= 0.5 else "NEGATIVE"
    confidence = prediction if sentiment == "POSITIVE" else (1 - prediction)
    
    return {
        "sentiment": sentiment,
        "confidence": float(confidence),
        "probability": float(prediction)
    }

# Create confidence gauge
def create_gauge(confidence, sentiment):
    """Create a gauge chart for confidence visualization"""
    color = "#28a745" if sentiment == "POSITIVE" else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence", 'font': {'size': 24}},
        number = {'suffix': "%", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff9c4'},
                {'range': [75, 100], 'color': '#e8f5e9'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Main app
def main():
    # Load model
    model, tokenizer, config = load_model_and_tokenizer()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure the model files are in the './models' directory.")
        return
    
    # Header
    st.title("😊 Sentiment Analysis App")
    st.markdown("### Analyze the sentiment of your text in real-time!")
    st.markdown("---")
    
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
        1. Enter your text in the input box
        2. Click 'Analyze Sentiment'
        3. View the results instantly!
        
        **Tip:** You can also try the sample texts below!
        """)
        
        st.markdown("---")
        st.header("📝 Sample Texts")
        samples = {
            "Positive 1": "I love this product! It's amazing!",
            "Negative 1": "This is terrible and broken",
            "Neutral": "Not bad, but could be better",
            "Positive 2": "Best purchase I've ever made!",
            "Negative 2": "Worst experience ever"
        }
        
        selected_sample = st.selectbox("Choose a sample:", [""] + list(samples.keys()))
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        if selected_sample and selected_sample in samples:
            default_text = samples[selected_sample]
        else:
            default_text = ""
        
        user_input = st.text_area(
            "Enter your text here:",
            value=default_text,
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter any text to analyze its sentiment"
        )
        
        # Analyze button
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### Quick Stats")
        st.metric("Max Text Length", f"{config['max_seq_length']} tokens")
        st.metric("Vocabulary Size", f"{config['max_vocab_size']:,}")
        st.metric("Model Type", "BiLSTM")
    
    # Results section
    if analyze_button and user_input:
        with st.spinner("Analyzing sentiment..."):
            # Simulate processing time for better UX
            time.sleep(0.5)
            
            # Get prediction
            result = predict_sentiment(
                user_input, 
                model, 
                tokenizer, 
                config['max_seq_length']
            )
            
            if result:
                st.markdown("---")
                st.markdown("## 📈 Analysis Results")
                
                # Create two columns for results
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    # Sentiment display
                    emoji = "😊" if result['sentiment'] == "POSITIVE" else "😞"
                    color_class = "sentiment-positive" if result['sentiment'] == "POSITIVE" else "sentiment-negative"
                    
                    st.markdown(f"""
                    <div class="confidence-box">
                        <h2 style="text-align: center;">{emoji}</h2>
                        <h3 class="{color_class}" style="text-align: center;">
                            {result['sentiment']}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional metrics
                    st.metric("Raw Probability", f"{result['probability']:.4f}")
                
                with res_col2:
                    # Confidence gauge
                    fig = create_gauge(result['confidence'], result['sentiment'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                st.markdown("---")
                st.markdown("### 💡 Interpretation")
                
                if result['confidence'] > 0.9:
                    confidence_text = "Very High"
                    interpretation = "The model is very confident about this prediction."
                elif result['confidence'] > 0.75:
                    confidence_text = "High"
                    interpretation = "The model is quite confident about this prediction."
                elif result['confidence'] > 0.6:
                    confidence_text = "Moderate"
                    interpretation = "The model has moderate confidence in this prediction."
                else:
                    confidence_text = "Low"
                    interpretation = "The model is uncertain about this prediction. The text might be neutral or ambiguous."
                
                st.info(f"**Confidence Level:** {confidence_text} ({result['confidence']*100:.2f}%) - {interpretation}")
    
    elif analyze_button:
        st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with ❤️ using Streamlit & TensorFlow</p>
        <p style="font-size: 12px;">Model trained on Sentiment140 dataset with 1.6M tweets</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()