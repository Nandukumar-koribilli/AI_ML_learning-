import streamlit as st
import torch
from transformers import pipeline
import time

# --- 1. PAGE CONFIGURATION (Make it look professional) ---
st.set_page_config(page_title="AI Hackathon Demo", page_icon="🤖", layout="centered")

st.title("🤖 AI Problem Solver")
st.markdown("### Powered by RTX 4050 & PyTorch")

# --- 2. LOAD THE MODEL (The "Brain") ---
# @st.cache_resource keeps the model in memory so it doesn't reload every time you click a button.
@st.cache_resource
def load_model():
    print("Loading model to GPU...")
    # 'device=0' tells it to use your RTX 4050.
    # We are using a pre-trained Sentiment Analysis model as a placeholder.
    # You can change "sentiment-analysis" to "summarization", "text-generation", etc.
    return pipeline("sentiment-analysis", device=0)

with st.spinner("Waking up the GPU..."):
    model = load_model()

# --- 3. THE INPUT (Where the user types) ---
user_input = st.text_area("Enter text to analyze:", height=150, placeholder="Type something here...")

# --- 4. THE MAGIC (Triggering the AI) ---
if st.button("Analyze Text", type="primary"):
    if user_input:
        start_time = time.time()
        
        # Run the model
        result = model(user_input)
        
        end_time = time.time()
        
        # --- 5. THE OUTPUT (Presenting Neatly) ---
        st.success("Analysis Complete!")
        
        # Create columns to organize results nicely
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Label", result[0]['label'])
        
        with col2:
            st.metric("Confidence Score", f"{result[0]['score'] * 100:.2f}%")
            
        st.caption(f"⏱️ Processed in {end_time - start_time:.4f} seconds on GPU.")
        
    else:
        st.warning("⚠️ Please enter some text first.")

# --- 6. SIDEBAR (Extra details for judges) ---
with st.sidebar:
    st.header("Project Info")
    st.info("This model runs locally on NVIDIA RTX 4050.")
    st.markdown("---")
    st.write("Built for **Hackathon 2025**")

    