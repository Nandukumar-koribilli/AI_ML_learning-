import streamlit as st
import pandas as pd
import torch
from transformers import pipeline

# --- PAGE SETUP ---
st.set_page_config(page_title="Dataset Analyzer", layout="wide")
st.title("📂 Hackathon Data Cruncher")
st.markdown("### Upload a CSV -> Run AI on it -> Download Results")

# --- 1. SETUP THE GPU BRAIN ---
@st.cache_resource
def load_ai():
    # We use device=0 for your RTX 4050
    # "sentiment-analysis" is the default. You can change this to "summarization" etc.
    return pipeline("sentiment-analysis", device=0)

model = load_ai()

# --- 2. FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Show the first few rows so you can check data
    st.write("### 1. Data Preview:")
    st.dataframe(df.head())

    # --- 3. COLUMN SELECTOR ---
    # Ask the user: Which column contains the text to analyze?
    # We filter for columns that are likely text (object/string types)
    text_columns = df.select_dtypes(include=['object']).columns
    text_column = st.selectbox("Select the column to analyze:", text_columns)

    if st.button("🚀 Run AI Analysis"):
        st.info(f"Processing {len(df)} rows on RTX 4050... this will be fast!")
        
        # --- 4. THE PROCESSING LOOP ---
        my_bar = st.progress(0)
        
        # Convert column to list for processing
        data_to_process = df[text_column].astype(str).tolist()
        
        # Run the model (Batching is automatic in pipelines usually, but this is simple)
        try:
            ai_output = model(data_to_process)
            
            # Extract labels and scores
            labels = [x['label'] for x in ai_output]
            scores = [x['score'] for x in ai_output]
            
            # Add results back to the DataFrame
            df['AI_Label'] = labels
            df['AI_Confidence'] = scores
            
            my_bar.progress(100)
            st.success("Analysis Complete!")
            
            # --- 5. RESULTS & DOWNLOAD ---
            st.write("### 2. Results:")
            st.dataframe(df.head())
            
            # Allow user to download the new CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Download Results as CSV",
                data=csv,
                file_name='ai_results.csv',
                mime='text/csv',
            )
        except Exception as e:
            st.error(f"Error processing data: {e}")