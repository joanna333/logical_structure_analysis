# pages/analysis.py 
import streamlit as st
import pandas as pd
from utils.prediction import predict_sentence

import re

def split_sentences_regex(text):
    # Clean the text
    text = re.sub(r'[\n\r]', ' ', text)  # Remove newlines
    text = re.sub(r'["\']', '', text)     # Remove quotes
    text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
    
    # More aggressive pattern that looks for sentence endings
    #pattern = r'[.!?]+[\s]+|[.!?]+$'
    pattern = r'[.]'
    # Split and clean resulting sentences
    sentences = [s.strip() for s in re.split(pattern, text) if s]
    
    # Filter out empty strings but keep sentences that don't start with capitals
    return [s for s in sentences if len(s) > 0]

def split_sentences_with_abbrev(text):
    # Common abbreviations to ignore
    abbreviations = {'mr.', 'mrs.', 'dr.', 'sr.', 'jr.', 'vs.', 'e.g.', 'i.e.', 'etc.'}
    
    # Split initially by potential sentence endings
    parts = text.split('. ')
    sentences = []
    current = parts[0]
    
    for part in parts[1:]:
        # Check if the previous part ends with an abbreviation
        ends_with_abbrev = any(current.lower().endswith(abbr) for abbr in abbreviations)
        
        if ends_with_abbrev:
            current = current + '. ' + part
        else:
            sentences.append(current)
            current = part
            
    sentences.append(current)
    return sentences

def show_analysis():
    st.title("Text Analysis")
    st.write("Use this section to analyze the logical structure of your text.")
    
    try:
        if 'model' not in st.session_state:
            st.error("Please initialize the model from the home page first.")
            return
            
        model = st.session_state.model
        label_encoder = st.session_state.label_encoder
        tokenizer = st.session_state.tokenizer
        
        # Text input section
        st.header("Analyze Your Text")
        user_text = st.text_area("Enter your text here (multiple sentences allowed):", height=150)
        
        if st.button("Analyze"):
            if user_text:
                # Split and analyze sentences
                sentences = split_sentences_regex(user_text)
                
                st.subheader("Analysis Results:")
                for i, sentence in enumerate(sentences, 1):
                    with st.container():
                        label, confidence = predict_sentence(
                            model, sentence, tokenizer, label_encoder
                        )
                        if label not in ("Unknown", "Error"):
                            st.write("---")
                            st.write(f"**Sentence:** {sentence}")
                            st.write(f"**Predicted:** {label}")
                            st.progress(confidence)
            else:
                st.warning("Please enter some text to analyze.")
        
        # Example Analysis Section
        st.header("Example Analysis")
        show_examples = st.checkbox("Show example analysis", key='show_examples')
        
        if show_examples:
            try:
                df = pd.read_csv('data/raw/history_01.csv')
                for sentence in df['Sentence'].head(5):  # Limit to 5 examples
                    with st.container():
                        label, confidence = predict_sentence(
                            model, sentence, tokenizer, label_encoder
                        )
                        if label not in ("Unknown", "Error"):
                            st.write("---")
                            st.write(f"**Sentence:** {sentence}")
                            st.write(f"**Predicted:** {label}")
                            st.progress(confidence)
            except FileNotFoundError:
                st.warning("Example file not found. Please check the data path.")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    show_analysis()