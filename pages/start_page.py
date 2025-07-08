import streamlit as st

def show_start_page():
    st.markdown("""
        <div class="typing-container">
            <div class="typing-text">Welcome to Kidney Tumor Detection</div>
        </div>
        <style>
        .typing-container {
            height: 85vh;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            text-align: center;
        }
        .typing-text {
            font-size: 42px;
            font-weight: bold;
            color: #2c3e50;
            border-right: 3px solid rgba(0,0,0,0.75);
            white-space: nowrap;
            overflow: hidden;
            width: 0;
            animation: typing 4s steps(40, end) forwards, blink .75s step-end infinite;
        }
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        @keyframes blink {
            from, to { border-color: transparent }
            50% { border-color: black }
        }
        </style>
    """, unsafe_allow_html=True)

    col = st.columns(3)[1]
    with col:
        if st.button("👉 Let's Go", use_container_width=True):
            st.session_state.page = "dashboard"
