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
        st.write("This is a test")
        
        if st.button("Click me"):
            st.write("Button clicked!")
        
        # Initialize collector
        collector = WikipediaCollector()
        logger.info("WikipediaCollector initialized")
        
        # Search interface
        st.subheader("Search Topics")
        topic_input = st.text_area(
            "Enter medical topics (one per line)", 
            placeholder="Example:\nAnatomy\nPhysiology\nNeurology"
        )
        
        if st.button("Collect Articles"):
            if topic_input:
                topics = [t.strip() for t in topic_input.split("\n") if t.strip()]
                
                with st.spinner("Collecting articles..."):
                    articles = collector.collect_articles(topics)
                
                if articles:
                    st.success(f"Collected {len(articles)} articles")
                    
                    df = pd.DataFrame(list(articles.items()), 
                                    columns=['Topic', 'Content'])
                    
                    st.subheader("Article Preview")
                    for topic, content in articles.items():
                        with st.expander(topic):
                            st.text(content[:500] + "...")
                    
                    st.download_button(
                        label="Download Articles (CSV)",
                        data=df.to_csv(index=False),
                        file_name="tms_articles.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No articles found. Please try different topics.")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()