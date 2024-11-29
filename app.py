# app.py
import streamlit as st


def main():

    # App title and description
    st.set_page_config(page_title="TMS Prep Article Collector")
    st.title("TMS Article Collector")
    st.write("This is a simple test page")
    
    # Basic button test
    if st.button("Test Button"):
        st.write("Button clicked!")


if __name__ == "__main__":
    main()