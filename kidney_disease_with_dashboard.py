import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime
from ultralytics import YOLO

# ------------------- Page Config -------------------
st.set_page_config(page_title="Kidney Tumor Detection", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "start"

# ------------------- Styling -------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#6D6D74, #EFECE5, #FFFFFF, #EFECE5);
        font-family: 'Segoe UI', sans-serif;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(to bottom, #f7f8fa, #dfe4ea);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
        transition: transform 0.2s;
    }
    .card:hover {
        transform: scale(1.02);
    }
    .card h3 {
        margin: 0;
        font-size: 20px;
        color: #2c3e50;
    }
    .card p {
        margin: 0;
        font-size: 18px;
        color: #34495e;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Forecast Image Generation -------------------
def simulate_forecast_images(image_np, tumor_boxes, scale_factor, color):
    forecast = image_np.copy()
    for box in tumor_boxes:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = int((x2 - x1) * scale_factor), int((y2 - y1) * scale_factor)
        nx1, ny1 = cx - w // 2, cy - h // 2
        nx2, ny2 = cx + w // 2, cy + h // 2
        cv2.rectangle(forecast, (nx1, ny1), (nx2, ny2), color, 3)
    return forecast

# ------------------- Image Analysis -------------------
def analyze_image(img_pil):
    model = YOLO("runs/detect/kidne_model3/weights/best.pt")
    img_np = np.array(img_pil.convert("RGB"))
    annotated = img_np.copy()
    results = model(img_np)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    names = model.names

    kidneys, tumors = [], []
    stage_counts = [0, 0, 0]
    h, w = img_np.shape[:2]

    for box, cls_id in zip(boxes, class_ids):
        label = names[cls_id].lower()
        if label == "kidney":
            kidneys.append(box)
        elif label == "tumor":
            tumors.append(box)

    tumor_matched = [False] * len(kidneys)
    overlay = annotated.copy()
    alpha = 0.3

    for tumor_box in tumors:
        x1, y1, x2, y2 = map(int, tumor_box[:4])
        tumor_area = (x2 - x1) * (y2 - y1)
        tumor_percent = (tumor_area / (w * h)) * 100

        for i, kidney_box in enumerate(kidneys):
            if is_inside(tumor_box, kidney_box) or boxes_intersect(tumor_box, kidney_box):
                tumor_matched[i] = True
                break

        stage = (
            "Stage 1" if tumor_percent < 20 else
            "Stage 2" if tumor_percent < 50 else
            "Stage 3"
        )
        if stage == "Stage 1": stage_counts[0] += 1
        if stage == "Stage 2": stage_counts[1] += 1
        if stage == "Stage 3": stage_counts[2] += 1

        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)

    for i, kidney_box in enumerate(kidneys):
        x1, y1, x2, y2 = map(int, kidney_box[:4])
        color = (0, 0, 255) if tumor_matched[i] else (0, 255, 0)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

    save_patient_record(stage_counts)

    return annotated, len(kidneys), len(tumors), stage_counts, tumors

# ------------------- Save Data -------------------
def save_patient_record(stage_counts):
    status = "Normal" if sum(stage_counts) == 0 else "Tumor Detected"
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage_1": stage_counts[0],
        "stage_2": stage_counts[1],
        "stage_3": stage_counts[2],
        "status": status
    }

    file_path = "patient_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])

    df.to_csv(file_path, index=False)

# ------------------- Utilities -------------------
def is_inside(boxA, boxB):
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB
    return Ax1 >= Bx1 and Ay1 >= By1 and Ax2 <= Bx2 and Ay2 <= By2

def boxes_intersect(boxA, boxB):
    Ax1, Ay1, Ax2, Ay2 = boxA
    Bx1, By1, Bx2, By2 = boxB
    return not (Ax2 < Bx1 or Ax1 > Bx2 or Ay2 < By1 or Ay1 > By2)

# ------------------- Charts -------------------
def plot_stage_chart():
    file_path = "patient_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        stage_1_total = df["stage_1"].sum()
        stage_2_total = df["stage_2"].sum()
        stage_3_total = df["stage_3"].sum()
        df["tumor_present"] = df[["stage_1", "stage_2", "stage_3"]].sum(axis=1) > 0
        tumor_patient_count = df["tumor_present"].sum()
        normal_patient_count = len(df[df["status"] == "Normal"])
    else:
        stage_1_total = stage_2_total = stage_3_total = tumor_patient_count = normal_patient_count = 0

    x_categories = ["Stage 1", "Stage 2", "Stage 3", "Total Tumor", "Normal"]
    tumor_counts = [stage_1_total, stage_2_total, stage_3_total, tumor_patient_count, None]
    normal_counts = [None, None, None, None, normal_patient_count]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_categories, y=tumor_counts, mode="lines+markers+text", name="Tumor Patients",
                             line=dict(color='red', width=3),
                             text=[f"{v} Patients" if v is not None else "" for v in tumor_counts],
                             textposition="top center"))
    fig.add_trace(go.Scatter(x=x_categories, y=normal_counts, mode="lines+markers+text", name="Normal Patients",
                             line=dict(color='green', width=3, dash='dot'),
                             text=[f"{v} Patients" if v is not None else "" for v in normal_counts],
                             textposition="top center"))

    fig.update_layout(title="Tumor vs Normal Patient Line Chart",
                      xaxis_title="Category", yaxis_title="Number of Patients", showlegend=True)
    return fig

def plot_bar_chart(stage_counts):
    fig = go.Figure([go.Bar(
        x=["Stage 1", "Stage 2", "Stage 3"],
        y=stage_counts,
        marker_color=['#BDC3C7', '#95A5A6', '#7F8C8D']
    )])
    fig.update_layout(title="Tumor Stage Count (Current Upload)", xaxis_title="Stage", yaxis_title="Count")
    return fig

# ------------------- Suggestions -------------------
def generate_suggestions(stage_counts):
    tips = []
    if sum(stage_counts) == 0:
        tips.append("✅ No tumors detected — kidneys appear healthy.")
    if stage_counts[0] > 0:
        tips.append("🟢 Stage 1 tumor(s) detected — early-stage. Schedule checkup.")
    if stage_counts[1] > 0:
        tips.append("🟠 Stage 2 tumor(s) found — seek medical opinion.")
    if stage_counts[2] > 0:
        tips.append("🔴 Stage 3 tumor(s) detected — consult oncologist urgently.")
    if all(x > 0 for x in stage_counts):
        tips.append("🧠 Multiple tumor stages found — seek comprehensive care.")
    return tips

# ------------------- UI Pages -------------------
def show_start_page():
    st.markdown("""
        <style>
            .center-container {
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 80vh;
                text-align: center;
            }
            .typing-text {
                overflow: hidden;
                white-space: nowrap;
                border-right: .15em solid orange;
                animation: typing 3s steps(40, end), blink-caret .75s step-end infinite;
                font-size: 32px;
                font-weight: bold;
                color: #2c3e50;
                font-family: 'Segoe UI', sans-serif;
            }
            @keyframes typing {
                from { width: 0 }
                to { width: 100% }
            }
            @keyframes blink-caret {
                from, to { border-color: transparent }
                50% { border-color: orange }
            }
            .lets-go-button {
                position: absolute;
                bottom: 60px;
                text-align: center;
                width: 100%;
            }
        </style>
        <div class="center-container">
            <div class="typing-text">Welcome to Kidney Tumor Detection</div>
        </div>
        <div class="lets-go-button">
    """, unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("👉 Let's Go", use_container_width=True):
            st.session_state.page = "dashboard"
    st.markdown("</div>", unsafe_allow_html=True)

def show_dashboard():
    st.markdown('<h2 style="text-align:center; color:#2c3e50;">🧬 Kidney Tumor Detection Dashboard</h2>', unsafe_allow_html=True)



    uploaded_image = st.file_uploader("📷 Upload Kidney CT Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img_pil = Image.open(uploaded_image)
        annotated_img, kidney_count, tumor_count, stage_counts, tumors = analyze_image(img_pil)

        card_cols = st.columns(5)
        titles = ["Kidneys Detected", "Tumors Detected", "📊 Stage 1", "📊 Stage 2", "📊 Stage 3"]
        values = [kidney_count, tumor_count, *stage_counts]
        for col, title, val in zip(card_cols, titles, values):
            with col:
                st.markdown(f'<div class="card"><h3>{title}</h3><p>{val}</p></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_pil, caption="🩻 Original CT Image", use_container_width=True)

        with col2:
            st.image(annotated_img, caption="🧠 Annotated Tumor Detection", use_container_width=True)

        if tumor_count > 0:
            st.markdown("### 🔮 Tumor Growth Forecast")
            forecast_2w = simulate_forecast_images(annotated_img, tumors, 1.2, (255, 165, 0))
            forecast_1m = simulate_forecast_images(annotated_img, tumors, 1.4, (128, 0, 128))
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.image(forecast_2w, caption="🟠 Forecast After 2 Weeks", use_container_width=True)

            with col_f2:
                st.image(forecast_1m, caption="🟣 Forecast After 1 Month", use_container_width=True)

            st.markdown("### 🧠 Smart Suggestions")
            for suggestion in generate_suggestions(stage_counts):
                st.markdown(f"- {suggestion}")

        chart1, chart2 = st.columns(2)
        with chart1:
            st.plotly_chart(plot_stage_chart(), use_container_width=True)
        with chart2:
            st.plotly_chart(plot_bar_chart(stage_counts), use_container_width=True)

    # ✅ Table of All Past Patient Records
    st.markdown("## 📋 Previous Patient Records")
    file_path = "patient_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = df.sort_values(by="timestamp", ascending=False)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No patient data available yet.")

# ------------------- Router -------------------
if st.session_state.page == "start":
    show_start_page()
else:
    show_dashboard()
