# pages/about.py
import streamlit as st

def show_about():
    st.title("About This Project")
    
    # Project Overview
    st.header("Logical Structure Analysis")
    st.write("""
    This application analyzes the logical structure of scientific text passages using 
    a BiLSTM-BERT model. It can identify 22 different types of logical relationships
    between sentences.
    """)
    
    # Model Details
    st.header("Model Architecture")
    st.write("""
    - Base: BioBERT (biomedical domain pre-trained BERT)
    - Architecture: Bidirectional LSTM with Attention
    - Training Data: 14,878 labeled sentences
    - Accuracy: 72.53% on validation set
    """)
    
    # Technologies
    st.sidebar.header("Technologies Used")
    tech_stack = {
        "Frontend": "Streamlit",
        "ML Framework": "PyTorch",
        "Language Model": "BioBERT",
        "Data Processing": "Pandas, NumPy"
    }
    for tech, tool in tech_stack.items():
        st.sidebar.write(f"**{tech}:** {tool}")
    
    # Contact/Links
    st.header("Contact & Resources")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[GitHub Repository](your-repo-link)")
    with col2:
        st.markdown("[Documentation](your-docs-link)")

if __name__ == "__main__":
    show_about()