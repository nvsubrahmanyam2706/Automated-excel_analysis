# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from llm_helper import configure_gemini, suggest_algorithms, generate_code_for_algorithm
from report_generator import generate_word_report

# ML Libraries
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

# ---------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="AI Excel Analyzer (Gemini)", layout="wide")
st.title("🤖 Gemini-Driven Excel Data Analysis (with Agents)")

# ---------------------------------------------------------
# Step 1: Gemini API Key
# ---------------------------------------------------------
api_key = st.text_input("🔑 Enter Gemini API Key:", type="password")
if not api_key:
    st.stop()
configure_gemini(api_key)

# ---------------------------------------------------------
# Step 2: Upload Excel
# ---------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Excel file", type=["xls", "xlsx"])
if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)
df = df.reset_index(drop=True)
st.write("### Dataset Preview")
st.dataframe(df.head())
st.write("**Shape:**", df.shape)
st.write("**Missing values:**")
st.write(df.isnull().sum())

# ---------------------------------------------------------
# Step 3: User Prompt
# ---------------------------------------------------------
prompt = st.text_input("💬 Enter your analysis goal or question")
if not prompt:
    st.stop()

# Initialize persistent session state
if "results_text" not in st.session_state:
    st.session_state.results_text = ""
if "visualizations" not in st.session_state:
    st.session_state.visualizations = []
if "algo_executed" not in st.session_state:
    st.session_state.algo_executed = False

# ---------------------------------------------------------
# Step 4: Gemini Algorithm Suggestion
# ---------------------------------------------------------
st.info("Generating algorithm suggestions using Gemini...")
algos = suggest_algorithms(prompt, df.head())  # using only df.head() for performance
st.write("### 🧠 Gemini Suggested Algorithms")
cols = st.columns(len(algos))

for i, algo in enumerate(algos):
    if cols[i].button(algo):
        st.session_state.results_text = ""  # reset before each run
        st.session_state.visualizations = []
        st.info(f"Gemini is executing {algo}... Please wait ⏳")

        # Generate Gemini's code dynamically (still passes df.head for speed)
        code = generate_code_for_algorithm(prompt, df.head(), algo)

        # --- Capture Plots ---
        captured_plots = []

        def capture_pyplot(fig=None):
            """Intercept plt calls from Gemini for saving visualizations."""
            buf = BytesIO()
            if fig is None:
                fig = plt.gcf()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            captured_plots.append(buf.read())
            st.image(buf)
            plt.close(fig)

        # --- Preprocess Data ---
        df_cleaned = df.copy()
        for col in df_cleaned.select_dtypes(include=np.number).columns:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        for col in df_cleaned.select_dtypes(exclude=np.number).columns:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        df_cleaned.dropna(how="all", inplace=True)
        df_cleaned.reset_index(drop=True, inplace=True)

        # --- Local Environment ---
        local_env = {
            "df": df_cleaned.copy(),
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
            "xgb": xgb,
        }

        local_env["st"].pyplot = capture_pyplot

        # --- Execute Gemini Code Safely ---
        try:
            exec(code, local_env)
            st.success(f"{algo} executed successfully! ✅")
            st.session_state.results_text += f"\n\n### {algo} Analysis\nGemini successfully executed {algo}.\n"
            st.session_state.visualizations.extend(captured_plots)
            st.session_state.algo_executed = True

        except KeyError as ke:
            st.warning(f"⚠️ Gemini tried to access an invalid column: {ke}. Please check your dataset columns.")
        except ValueError as ve:
            if "length" in str(ve).lower():
                st.warning("⚠️ Gemini code attempted to combine mismatched arrays. Some charts may not align, but analysis continues.")
            else:
                st.error(f"Gemini execution error: {ve}")
        except Exception as e:
            st.error(f"Unexpected error running Gemini code: {e}")
            st.session_state.algo_executed = False

# ---------------------------------------------------------
# Step 4.5: KPI Dashboard (Toggle)
# ---------------------------------------------------------
if st.session_state.algo_executed:
    st.write("### 📊 Dashboard Mode")
    enable_dashboard = st.toggle("Enable Gemini KPI Dashboard", value=False, key="dashboard_toggle")

    if enable_dashboard:
        from dashboard_agent import generate_dashboard
        st.info("🧠 Gemini Dashboard Agent is analyzing your data...")

        try:
            kpi_text, dashboard_images = generate_dashboard(df, prompt)

            if kpi_text.strip():
                st.subheader("📈 Gemini-Generated KPIs")
                st.markdown(kpi_text)
                st.session_state.results_text += f"\n\n### Gemini Dashboard KPIs\n{kpi_text}"

            if dashboard_images:
                st.subheader("📊 KPI Dashboard Visualizations")
                cols = st.columns(2)
                for idx, img_bytes in enumerate(dashboard_images):
                    with cols[idx % 2]:
                        st.image(img_bytes)
                st.session_state.visualizations.extend(dashboard_images)

            st.success("✅ Dashboard generated successfully!")
        except Exception as e:
            st.error(f"Dashboard Agent Error: {e}")

# ---------------------------------------------------------
# Step 5: Generate Word Report
# ---------------------------------------------------------
st.write("### 📄 Generate Report")
if st.button("📥 Download Word Report"):
    if not st.session_state.results_text and not st.session_state.visualizations:
        st.warning("⚠️ Run at least one algorithm before generating the report.")
    else:
        report_buffer = generate_word_report(
            df=df,
            prompt=prompt,
            results_text=st.session_state.results_text,
            visualizations=st.session_state.visualizations,
        )
        st.download_button(
            label="📄 Download Gemini Word Report",
            data=report_buffer,
            file_name="AI_Excel_Report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
