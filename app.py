# app.py
import streamlit as st
import pandas as pd
from src.data.data_collector import WikipediaCollector
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # App title and description
        st.set_page_config(page_title="TMS Prep Article Collector")
        st.title("TMS Article Collector")
        st.write("This is a simple test page")
        
        # Basic button test
        if st.button("Test Button"):
            st.write("Button clicked!")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()