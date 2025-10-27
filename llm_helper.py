# llm_helper.py
import google.generativeai as genai
import pandas as pd

def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)

def suggest_algorithms(prompt: str, df_head: pd.DataFrame):
    model = genai.GenerativeModel("gemini-2.5-flash")
    query = f"""
You are a data science assistant.
Dataset sample:
{df_head.to_string(index=False)}

User prompt: "{prompt}"

Suggest exactly 3 machine learning algorithms suitable for this data and prompt.
Return only algorithm names separated by commas.
"""
    response = model.generate_content(query)
    algos = [x.strip() for x in response.text.split(",")[:3]]
    return algos

def generate_code_for_algorithm(prompt: str, df_head: pd.DataFrame, selected_algo: str):
    model = genai.GenerativeModel("gemini-2.5-flash")
    query = f"""
You are a Python data science assistant.
Dataset sample:
{df_head.to_string(index=False)}

User prompt: "{prompt}"
Selected algorithm: "{selected_algo}"

Generate Python code that:
1. Resets dataframe index before any filtering to avoid unalignable boolean index errors.
2. Handles missing values (numerical -> median, categorical -> mode).
3. Ensures X and y have equal length before fitting.
4. Fits the ML model properly.
5. Generates 2–4 visualizations (histograms, scatterplots, etc.).
6. Prints metrics and st.write() insights.
7. Uses:
   st.pyplot(plt.gcf())
   plt.clf()
   instead of plt.show().
Return only executable Python code.
"""
    response = model.generate_content(query)
    code = response.text.strip()
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:-1])
    return code
