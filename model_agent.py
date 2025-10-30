
# model_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from llm_helper import generate_code_for_algorithm


def run_model_analysis(df: pd.DataFrame, prompt: str, selected_algo: str):
    """
    Runs the Gemini-generated ML model code, captures insights & visualizations.
    Returns:
        (results_text, visuals)
    """
    results_text = ""
    visuals = []

    # Step 1: Clean data before passing to Gemini
    df_cleaned = df.copy()
    for col in df_cleaned.select_dtypes(include=np.number).columns:
        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    for col in df_cleaned.select_dtypes(exclude=np.number).columns:
        if df_cleaned[col].isnull().any():
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    df_cleaned.dropna(how="all", inplace=True)
    df_cleaned.reset_index(drop=True, inplace=True)

    # Step 2: Ask Gemini to generate code for the selected algorithm
    try:
        code = generate_code_for_algorithm(prompt, df_cleaned.head(), selected_algo)
    except Exception as e:
        return f"Error generating Gemini code: {e}", visuals

    if not isinstance(code, str) or len(code.strip()) == 0:
        return "Gemini did not return valid executable code.", visuals

    # Step 3: Capture visuals
    def capture_plot(fig=None):
        """Capture plt figures as image bytes."""
        if fig is None:
            fig = plt.gcf()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        visuals.append(buf.read())
        plt.close(fig)

    # Step 4: Safe local execution
    local_env = {
        "df": df_cleaned.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "st": None,  # Gemini code expects 'st', we replace functions manually
        "train_test_split": None,
        "results": [],
    }

    # Replace streamlit calls with local capture functions
    class DummySt:
        @staticmethod
        def write(x):
            nonlocal results_text
            results_text += str(x) + "\n"

        @staticmethod
        def pyplot(fig=None):
            capture_plot(fig)

    local_env["st"] = DummySt()

    # Step 5: Execute Gemini-generated model code
    try:
        exec(code, local_env)
    except Exception as e:
        results_text += f"\n⚠️ Error during execution: {e}\n"

    # Step 6: Return insights & captured visuals
    if not visuals:
        # fallback visualization if Gemini didn’t plot anything
        plt.figure(figsize=(6, 4))
        sns.heatmap(df_cleaned.corr(numeric_only=True), annot=True, cmap="coolwarm")
        capture_plot(plt.gcf())
        results_text += "\n(Generated correlation heatmap since no plots were produced by Gemini.)"

    return results_text.strip(), visuals
