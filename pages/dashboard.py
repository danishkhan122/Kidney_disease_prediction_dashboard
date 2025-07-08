import streamlit as st
from PIL import Image
from utils.analyzer import analyze_image
from utils.charts import plot_stage_chart, plot_bar_chart
from utils.suggestion import generate_suggestions

def show_dashboard():
    st.markdown('<h2 style="text-align:center; color:#2c3e50;">🧬 Kidney Tumor Detection Dashboard</h2>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("📷 Upload Kidney CT Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img_pil = Image.open(uploaded_image)
        annotated_img, kidney_count, tumor_count, stage_counts = analyze_image(img_pil)

        titles = ["Kidneys Detected", "Tumors Detected", "📊 Stage 1", "📊 Stage 2", "📊 Stage 3"]
        values = [kidney_count, tumor_count, *stage_counts]
        for col, title, val in zip(st.columns(5), titles, values):
            with col:
                st.markdown(f'<div class="card"><h3>{title}</h3><p>{val}</p></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_pil, caption="🩻 Original CT Image", use_column_width=True)
        with col2:
            st.image(annotated_img, caption="🧠 Annotated Image", use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_stage_chart(stage_counts), use_container_width=True)
        with col2:
            st.plotly_chart(plot_bar_chart(stage_counts), use_container_width=True)

        st.markdown("### 🧠 Smart Suggestions")
        for tip in generate_suggestions(stage_counts):
            st.markdown(f"- {tip}")
