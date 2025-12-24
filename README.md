# 🚀 AI Hackathon Super-App (RTX 4050 Edition)

## 📌 Project Overview
This project is an all-in-one AI Dashboard built for high-speed problem solving. It leverages a **local NVIDIA RTX 4050 GPU** to perform real-time analysis across three domains:
1.  **📝 NLP (Text):** Sentiment analysis, summarization, and text generation.
2.  **👁️ Computer Vision:** Zero-shot image classification (OpenAI CLIP).
3.  **📊 Tabular Data:** Auto-ML for predicting numbers/categories from Excel/CSV files.

---

## ⚙️ Quick Setup
* **Environment:** Conda (Python 3.10)
* **Frameworks:** PyTorch (CUDA 11.8), Streamlit, Transformers, Scikit-Learn.
* **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM).

### Installation (If moving to a new PC)
```bash
conda create -n hackathon python=3.10
conda activate hackathon
pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install pandas numpy matplotlib seaborn scikit-learn transformers datasets streamlit

```
### to run the py file 

```bash
& "C:\Users\Govin\anaconda3\envs\hackathon\python.exe" -m streamlit run main.py

```

