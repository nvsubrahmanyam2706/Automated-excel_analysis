# 🤖 Gemini AI Excel Analyzer

A Streamlit-based intelligent data analysis platform powered by **Google Gemini AI**.
It automatically analyzes uploaded Excel datasets, suggests suitable machine-learning algorithms, executes them dynamically, visualizes insights, and generates a complete **Word report** — all with a single click.

---

## 🚀 Key Features

* **Gemini-Powered Algorithm Suggestion**
  Automatically recommends 3 best-fit ML algorithms (classification, regression, clustering, or forecasting).

* **Dynamic Code Generation & Execution**
  Gemini writes and executes Python code for the selected algorithm on your dataset.

* **Automated KPI Dashboard**
  Generates KPIs, statistical summaries, and clean data visualizations.

* **Interactive Streamlit UI**
  Upload Excel → Select Goal → Choose Algorithm → View Results Instantly.

* **Word Report Generation**
  Exports analysis summary, KPIs, and charts into a professional `.docx` file.

---

## 🧩 Project Structure

```
├── app.py                # Main Streamlit app (UI + workflow)
├── llm_helper.py         # Handles Gemini API setup, suggestions, and code generation
├── dashboard_agent.py    # Auto-generates KPI dashboard and visualizations
├── model_agent.py        # (Optional) Isolates model execution logic for future extensions
├── report_generator.py   # Creates Word reports with results and plots
├── requirements.txt      # Python dependencies
```

---

## 🧠 Technology Stack

* **Language:** Python 3.10+
* **Framework:** Streamlit
* **LLM:** Google Gemini 2.5-Flash API
* **Libraries:**

  * `pandas`, `numpy`, `matplotlib`, `seaborn`
  * `scikit-learn`, `xgboost`, `prophet`, `statsmodels`
  * `python-docx` for report generation
  * `google-generativeai` for Gemini API

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/AI-Excel-Analyzer.git
cd AI-Excel-Analyzer
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up Google Gemini API

Obtain your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
and enter it in the Streamlit UI when prompted.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the displayed local URL (e.g., `http://localhost:8501`) in your browser.

---

## 🧾 How It Works

1. **Enter Gemini API Key** – authenticate to access Gemini-2.5-Flash model.
2. **Upload Excel File** – dataset preview, shape, and missing values are displayed.
3. **Enter Analysis Goal** – e.g., “Predict sales using available data”.
4. **Gemini Suggests Algorithms** – dynamically recommends 3 algorithms.
5. **Select Algorithm** – Gemini generates, executes, and visualizes the analysis.
6. **Enable KPI Dashboard (optional)** – view KPIs and trends from your dataset.
7. **Download Word Report** – get a polished `.docx` report with visuals.

---

## 📊 Example Use Cases

* Predictive modeling (Sales, Demand, Revenue)
* Customer segmentation
* Forecasting time-series data
* Exploratory data analysis for business insights

---

## 👨‍💻 Developers

**Nagadevara Veera Subrahmanyam , Kaduru Sujitha**
