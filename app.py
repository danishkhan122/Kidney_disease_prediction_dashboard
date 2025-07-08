import streamlit as st
from pages.start_page import show_start_page
from pages.dashboard import show_dashboard

st.set_page_config(page_title="Kidney Tumor Detection", layout="wide")

# Manage navigation state
if "page" not in st.session_state:
    st.session_state.page = "start"

if st.session_state.page == "start":
    show_start_page()
else:
    show_dashboard()
