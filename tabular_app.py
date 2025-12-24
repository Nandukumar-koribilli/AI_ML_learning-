import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Auto-ML Tabular", layout="wide")
st.title("📊 Universal Number Cruncher")
st.markdown("### Upload Data -> Select Target -> AI Trains Itself")

# --- 1. UPLOAD DATA ---
uploaded_file = st.file_uploader("Upload your CSV (Excel data)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:")
    st.dataframe(df.head())
    
    # --- 2. CLEAN DATA (Auto-Magic) ---
    # Convert text columns (like "Male/Female") into numbers (0/1) so AI can read them
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
            
    # --- 3. SELECT TARGET ---
    target_col = st.selectbox("What do you want to predict?", df.columns)
    
    if st.button("Train Model"):
        # Split data into X (Features) and y (Answer)
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Split into Training and Testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.write(f"Training on {len(X_train)} rows...")
        
        # --- 4. DETERMINE TYPE (Regression vs Classification) ---
        # If the target has few unique values (e.g., Yes/No), it's Classification.
        # If it has many (e.g., Price $50,000 - $500,000), it's Regression.
        unique_values = y.nunique()
        
        if unique_values < 20:
            st.info(f"Detected: Classification (Predicting categories)")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            
            # Evaluate
            acc = model.score(X_test, y_test)
            st.success(f"✅ Model Accuracy: {acc * 100:.2f}%")
            
        else:
            st.info(f"Detected: Regression (Predicting numbers)")
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            
            # Evaluate
            score = model.score(X_test, y_test)
            st.success(f"✅ Model Score (R2): {score:.4f}")

        # --- 5. FEATURE IMPORTANCE ---
        # Show what data mattered most
        st.write("### What mattered most?")
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        st.bar_chart(importance.set_index('Feature'))