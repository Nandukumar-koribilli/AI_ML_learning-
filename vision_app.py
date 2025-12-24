import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# --- PAGE SETUP ---
st.set_page_config(page_title="Vision AI", layout="wide")
st.title("👁️ Universal Image Classifier")
st.markdown("### Powered by RTX 4050 & OpenAI CLIP")

# --- 1. LOAD THE MODEL (Downloads ~600MB once) ---
@st.cache_resource
def load_vision_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, device

with st.spinner("Waking up the Vision AI..."):
    model, processor, device = load_vision_model()
    st.success("Vision Model Ready!")

# --- 2. USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

with col2:
    st.subheader("2. Define Categories")
    # This is the magic. You can type ANYTHING here.
    labels_input = st.text_input("What should I look for? (comma separated)", 
                                 value="a cat, a dog, a people, a plane")
    labels = [l.strip() for l in labels_input.split(",")]

# --- 3. THE AI PREDICTION ---
if uploaded_file and labels:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Identify Image"):
        # Process the image and text together
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
        
        # Move inputs to your RTX 4050
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Run Prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Calculate percentages
        probs = outputs.logits_per_image.softmax(dim=1)
        probs_list = probs.cpu().numpy()[0]
        
        # --- 4. SHOW RESULTS ---
        st.subheader("Analysis Results:")
        
        # Create a dictionary for the chart
        results = {label: float(prob) for label, prob in zip(labels, probs_list)}
        
        # Show bar chart
        st.bar_chart(results)
        
        # Announce the winner
        winner = labels[probs.argmax().item()]
        confidence = probs.max().item() * 100
        st.success(f"✅ I am **{confidence:.2f}%** sure this is: **{winner}**")