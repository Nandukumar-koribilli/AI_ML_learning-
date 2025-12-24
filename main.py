import streamlit as st
import pandas as pd
import torch
from PIL import Image
from transformers import pipeline, CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Hackathon Super-App", page_icon="🚀", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 AI Toolkit")
st.sidebar.info("Running on NVIDIA RTX 4050")
app_mode = st.sidebar.radio("Choose your AI Tool:", ["📝 Text Analyzer", "👁️ Vision (Images)", "📊 Data (Excel/Numbers)"])

# ==========================================
# MODE 1: TEXT ANALYSIS (NLP)
# ==========================================
if app_mode == "📝 Text Analyzer":
    st.title("📝 Intelligent Text Processor")
    st.markdown("Analyze reviews, emails, or news using Deep Learning.")

    # Task Selection
    task = st.selectbox("Select Task:", ["Sentiment Analysis", "Summarization", "Text Generation"])
    
    # Load Model Button (To save memory)
    if st.button("Load Text Model"):
        with st.spinner("Loading AI Brain to GPU..."):
            if task == "Sentiment Analysis":
                st.session_state['text_model'] = pipeline("sentiment-analysis", device=0)
            elif task == "Summarization":
                st.session_state['text_model'] = pipeline("summarization", device=0)
            elif task == "Text Generation":
                st.session_state['text_model'] = pipeline("text-generation", device=0)
        st.success(f"Loaded {task} Model!")

    # Input Area
    text_input = st.text_area("Enter Text Here:", height=150)
    
    if st.button("Analyze"):
        if 'text_model' in st.session_state:
            start = time.time()
            result = st.session_state['text_model'](text_input)
            end = time.time()
            
            st.write("### Result:")
            st.write(result)
            st.caption(f"⏱️ Processed in {end - start:.4f} seconds on GPU.")
        else:
            st.error("Please click 'Load Text Model' first!")

# ==========================================
# MODE 2: VISION (IMAGES)
# ==========================================
elif app_mode == "👁️ Vision (Images)":
    st.title("👁️ Zero-Shot Image Classifier")
    st.markdown("Identify objects in images without any training.")

    @st.cache_resource
    def load_clip():
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return model, processor, device

    with st.spinner("Loading Vision AI..."):
        model, processor, device = load_clip()

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    with col2:
        labels_input = st.text_input("Categories (comma separated)", value="cat, dog, car, plane")
        labels = [l.strip() for l in labels_input.split(",")]

    if uploaded_file and st.button("Identify Image"):
        image = Image.open(uploaded_file)
        st.image(image, width=300)
        
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = outputs.logits_per_image.softmax(dim=1)
        probs_list = probs.cpu().numpy()[0]
        
        st.bar_chart({label: float(prob) for label, prob in zip(labels, probs_list)})
        st.success(f"Winner: {labels[probs.argmax().item()]}")

# ==========================================
# MODE 3: TABULAR (NUMBERS/EXCEL)
# ==========================================
elif app_mode == "📊 Data (Excel/Numbers)":
    st.title("📊 Auto-ML Prediction Engine")
    st.markdown("Upload a CSV and automatically train a predictor.")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        df = df.fillna(0) # Fix empty cells
        st.dataframe(df.head(3))
        
        # Preprocessing (Convert text to numbers)
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col].astype(str))

        target = st.selectbox("Predict which column?", df.columns)
        
        if st.button("Train AI Model"):
            X = df.drop(columns=[target])
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Auto-detect type
            if y.nunique() < 20:
                model = RandomForestClassifier()
                type_name = "Classification"
            else:
                model = RandomForestRegressor()
                type_name = "Regression"
                
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            
            st.success(f"✅ Trained {type_name} Model. Accuracy/Score: {score:.4f}")
            
            st.write("### Top Factors:")
            imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            st.bar_chart(imp.set_index('Feature').sort_values('Importance', ascending=False))