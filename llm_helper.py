# utils/llm_helper.py

import google.generativeai as genai
import pandas as pd

# ---------- Configure Gemini ----------
def configure_gemini(api_key: str):
    """Configure Gemini LLM with the API key."""
    genai.configure(api_key=api_key)


# ---------- Suggest Algorithms ----------
def suggest_algorithms(prompt: str, df_head: pd.DataFrame):
    """Ask Gemini to suggest 3 algorithms based on the dataset + prompt."""
    model = genai.GenerativeModel("gemini-2.5-flash")
    query = f"""
You are a data science assistant.
The user uploaded a dataset (sample below):
{df_head.to_string()}

User prompt: "{prompt}"
Suggest exactly 3 machine learning algorithms suitable for this task.
Return only the algorithm names, separated by commas.
"""
    response = model.generate_content(query)
    algos = [x.strip() for x in response.text.split(",")[:3]]
    return algos


# ---------- Generate Code for Selected Algorithm ----------
def generate_code_for_algorithm(prompt: str, df_head: pd.DataFrame, selected_algo: str):
    """Ask Gemini to generate Python code ready for Streamlit execution."""
    model = genai.GenerativeModel("gemini-2.5-flash")

    query = f"""
You are a Python data science assistant. The user uploaded a dataset (sample below):
{df_head.to_string()}

User prompt: "{prompt}"
The user selected algorithm: "{selected_algo}"

Generate a complete Python code that:
1. Handles missing values (impute numerical with median, categorical with mode)
2. Splits data into train/test if applicable
3. Fits the selected ML algorithm
4. Generates plots (matplotlib / seaborn)
5. Prints metrics and textual insights

⚠️ IMPORTANT:
- Ensure NO NaN values remain before training or evaluation
- Replace all 'print()' statements with 'st.write()'
- Replace all 'plt.show()' statements with:
      st.pyplot(plt.gcf())
      plt.clf()
- The code should be fully executable in Streamlit
- Do NOT include explanations outside of code
- Only return Python code
"""
    response = model.generate_content(query)
    code = response.text.strip()

    # Strip code block markers if any
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:-1])
    return code
