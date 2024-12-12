import pandas as pd
from functools import lru_cache
import streamlit as st
from utils.prediction import (
    load_model_for_prediction,
    predict_sentence,
    get_device
)
from pages.about import show_about
from pages.analysis import show_analysis

@lru_cache(maxsize=1)
def load_model():
    """Load and cache model"""
    try:
        model_path = "models/model_epoch8_acc72.53.pt"
        model, label_encoder, tokenizer = load_model_for_prediction(model_path)
        device = get_device()
        model = model.to(device)
        return model, label_encoder, tokenizer
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

# Use in Streamlit
def init_model():
    """Initialize model for Streamlit"""
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.label_encoder, st.session_state.tokenizer = load_model()

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Analysis", "About"])
    
    if page == "Home":
        st.title("Text Classification App")
        try:
            init_model()
            st.write("Welcome to the Text Classification App!")
            st.write("Use the sidebar to navigate to different sections.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

    elif page == "Analysis":
        st.session_state.page = 'analysis'  # Add state
        show_analysis()
            
    elif page == "About":
        show_about()

if __name__ == "__main__":
    main()