# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from prophet import Prophet
import xgboost as xgb
import statsmodels.api as sm

# PDF / Word export
from docx import Document
from io import BytesIO

# Gemini helper
from llm_helper import configure_gemini, suggest_algorithms, generate_code_for_algorithm

st.set_page_config(page_title="AI Excel Analyzer (Gemini)", layout="wide")
st.title("🤖 Gemini-Driven Excel Data Analysis")

# ---------- Step 1: Gemini API Key ----------
api_key = st.text_input("🔑 Enter Gemini API Key:", type="password")
if not api_key:
    st.stop()
configure_gemini(api_key)

# ---------- Step 2: Upload Excel ----------
uploaded_file = st.file_uploader("📂 Upload Excel file", type=["xls", "xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())
    st.write("**Shape:**", df.shape)
    st.write("**Missing values:**")
    st.write(df.isnull().sum())

    # ---------- Step 3: User prompt ----------
    prompt = st.text_input("💬 Enter your analysis prompt")
    if prompt:
        st.info("Generating algorithm suggestions from Gemini...")
        algos = suggest_algorithms(prompt, df.head())
        st.write("### 🤖 Suggested Algorithms")
        cols = st.columns(len(algos))
        
        for i, algo in enumerate(algos):
            if cols[i].button(algo):
                st.info(f"Generating and executing {algo} code via Gemini...")

                # Generate Gemini code
                code = generate_code_for_algorithm(prompt, df.head(), algo)

                # Local environment for execution
                local_env = {
                    "df": df.copy(),
                    "pd": pd,
                    "st": st,
                    "plt": plt,
                    "sns": sns,
                    "np": np,
                    "train_test_split": train_test_split,
                    "LabelEncoder": LabelEncoder,
                    "OneHotEncoder": OneHotEncoder,
                    "ColumnTransformer": ColumnTransformer,
                    "Pipeline": Pipeline,
                    "DecisionTreeClassifier": DecisionTreeClassifier,
                    "RandomForestClassifier": RandomForestClassifier,
                    "KMeans": KMeans,
                    "DBSCAN": DBSCAN,
                    "AgglomerativeClustering": AgglomerativeClustering,
                    "Prophet": Prophet,
                    "sm": sm,
                    "xgb": xgb
                }

                # Execute Gemini-generated code
                try:
                    exec(code, local_env)
                    st.success(f"{algo} executed successfully! ✅")
                except Exception as e:
                    st.error(f"Error running Gemini code: {e}")

        # ---------- Step 4: Report Export ----------
        st.write("### 📄 Download Report")
        doc = Document()
        doc.add_heading("AI Excel Analysis Report", 0)
        doc.add_paragraph(f"User Prompt: {prompt}")
        doc.add_paragraph(f"Dataset Shape: {df.shape}")
        doc.add_heading("Missing Values:", level=1)
        doc.add_paragraph(df.isnull().sum().to_string())
        
        # Add first 5 rows preview
        doc.add_heading("Dataset Preview (first 5 rows):", level=1)
        doc.add_paragraph(df.head().to_string())
        
        # Save to BytesIO and provide download link
        buffer = BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Download Word Report",
            data=buffer,
            file_name="AI_Excel_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
