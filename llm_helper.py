# llm_helper.py
import google.generativeai as genai
import pandas as pd
import re


# ---------------------------------------------------------
# Configure Gemini
# ---------------------------------------------------------
def configure_gemini(api_key: str):
    """Configure Gemini LLM."""
    genai.configure(api_key=api_key)


# ---------------------------------------------------------
# Helper: Smart Data Sampler
# ---------------------------------------------------------
def _prepare_data_sample(df: pd.DataFrame) -> str:
    """
    Returns an intelligently sampled version of the dataset for Gemini.
    - If the dataset is small (< 100 rows), send the full data.
    - If it's large, send a random 20-row sample.
    """
    if len(df) <= 100:
        sample = df
    else:
        sample = df.sample(min(20, len(df)), random_state=42)
    return sample.to_string(index=False)


# ---------------------------------------------------------
# Algorithm Suggestion
# ---------------------------------------------------------
def suggest_algorithms(prompt: str, df: pd.DataFrame):
    """Use Gemini to suggest 3 suitable algorithms based on dataset & goal."""
    model = genai.GenerativeModel("gemini-2.5-flash")

    data_sample = _prepare_data_sample(df)
    query = f"""
You are an experienced Data Science AI Assistant specialized in analyzing structured data.

Below is a representative sample of the dataset:
{data_sample}

The user wants to achieve the following goal:
"{prompt}"

Analyze the dataset schema and suggest exactly 3 machine learning algorithms
that best fit this kind of problem (classification, regression, clustering, forecasting, etc.).

Return only the algorithm names separated by commas.
Example: Linear Regression, Random Forest, K-Means
"""
    response = model.generate_content(query)

    # Clean and normalize Gemini's output
    text = response.text.strip()
    text = re.sub(r'[\n;:/|]+', ',', text)  # replace weird separators
    text = re.sub(r'\s+and\s+', ',', text, flags=re.IGNORECASE)
    algos = [a.strip() for a in text.split(',') if a.strip()]

    return algos[:3] if len(algos) >= 3 else algos


# ---------------------------------------------------------
# Code Generation
# ---------------------------------------------------------
def generate_code_for_algorithm(prompt: str, df: pd.DataFrame, selected_algo: str):
    """Generate executable Python code for the selected algorithm using Gemini."""
    model = genai.GenerativeModel("gemini-2.5-flash")

    data_sample = _prepare_data_sample(df)
    query = f"""
You are a senior Python Data Scientist and Machine Learning Engineer.
You are given a dataset sample below and the user's analysis goal.

Dataset sample:
{data_sample}

User goal: "{prompt}"
Algorithm selected: "{selected_algo}"

Write Python code that:
1. Cleans missing data (median for numeric, mode for categorical)
2. Trains and evaluates the selected algorithm
3. Generates at least 4 insightful visualizations (matplotlib/seaborn)
4. Prints findings using st.write()
5. Uses st.pyplot(plt.gcf()); plt.clf() instead of plt.show()

Return only executable Python code — no extra explanation.
"""
    response = model.generate_content(query)
    code = response.text.strip()

    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:-1])

    return code
