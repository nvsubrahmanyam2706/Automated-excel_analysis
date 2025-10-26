# 🤖 AI Excel Analyzer (Gemini)

**AI Excel Analyzer (Gemini)** is an intelligent data analysis tool that integrates **Google Gemini LLM** with **Streamlit** to automatically analyze Excel datasets, suggest suitable machine learning algorithms, generate executable Python code, visualize results, and export a comprehensive report in Word format — all in one streamlined workflow.

---

## 🚀 Features

* 📂 Upload Excel datasets (`.xls`, `.xlsx`)
* 🧠 Automatically suggests 3 machine learning algorithms based on data and user prompt (via Gemini)
* 💻 Generates executable Python code for the selected algorithm
* 📊 Handles missing data, performs train/test splits, and visualizes results
* 📈 Runs popular algorithms like Decision Trees, Random Forest, XGBoost, Prophet, and Clustering
* 📄 Exports AI-generated insights and dataset details as a **Word report**
* 🔒 Secure Gemini API key input with on-demand configuration

---

## 🧩 Project Structure

```
AUTOMATED-EXCEL_ANALYSIS/
│
├── app.py                    # Streamlit main app file
├── requirements.txt          # Python dependencies
├── llm_helper.py             # Gemini integration helper  
functions
└── README.md                 # Project documentation
```

---

## ⚙️ Requirements

| Category          | Package                                       |
| ----------------- | --------------------------------------------- |
| **Core**          | Python 3.9+                                   |
| **Libraries**     | streamlit, pandas, numpy, seaborn, matplotlib |
| **ML Models**     | scikit-learn, xgboost, prophet, statsmodels   |
| **LLM**           | google-generativeai                           |
| **Report Export** | python-docx                                   |
| **Visualization** | matplotlib, seaborn                           |

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🏗️ Architecture

**1. User Interaction Layer (Streamlit UI)**

* Collects Gemini API key securely.
* Uploads Excel dataset.
* Takes user analysis prompt.
* Displays Gemini-suggested algorithms.
* Runs dynamically generated ML code.
* Exports insights as a Word report.

**2. LLM Processing Layer (utils/llm_helper.py)**

* Configures Gemini API.
* Sends dataset preview and user prompt to Gemini.
* Receives algorithm suggestions and auto-generates ML code.
* Ensures clean, Streamlit-compatible Python code.

**3. Data Processing & ML Layer (app.py)**

* Cleans data (handles missing values).
* Runs selected ML algorithm (classification, clustering, forecasting).
* Generates visualizations and metrics.
* Displays output in Streamlit dashboard.

**4. Report Generation Layer**

* Summarizes dataset info, missing values, and preview.
* Compiles insights into a downloadable **Word (.docx)** report.

---

## 🧠 How Gemini Works (Backend)

* The Gemini model (`gemini-2.5-flash`) is prompted with:

  * Dataset sample (via `df.head()`)
  * User’s text prompt
  * Instructions to suggest 3 ML algorithms or generate executable Python code
* Gemini responds with either:

  1. Algorithm suggestions (as a list)
  2. Python code that runs inside Streamlit
* The generated code executes dynamically using Python’s `exec()` inside a controlled local environment.

---

## 🔍 Implementation Progress

| Stage                  | Status         | Description                                       |
| ---------------------- | -------------- | ------------------------------------------------- |
| Gemini LLM Setup       | ✅ Completed    | API integration and configuration ready           |
| Excel Upload + Preview | ✅ Completed    | File upload and dataset inspection working        |
| Algorithm Suggestion   | ✅ Completed    | Gemini suggests top 3 relevant algorithms         |
| Code Generation        | ✅ Completed    | Gemini generates and executes working Python code |
| Data Cleaning          | ✅ Completed    | Handles NaN values using median/mode imputation   |
| Visualization          | ✅ Completed    | Supports plots via matplotlib/seaborn             |
| Report Export          | ✅ Completed    | Word (.docx) report generation implemented        |   

---

## 📈 Model Evaluation & Accuracy

* Gemini dynamically generates ML code, so model accuracy depends on:

  * Dataset quality
  * Chosen algorithm
  * Train/test split
* Accuracy metrics (e.g., accuracy score, RMSE, MAE) are calculated within the generated code.
* Typical results show **70–95% accuracy** for well-balanced datasets.

---

## 🧪 Data Augmentation

* Currently **not implemented** in this version.
* Planned for future update where Gemini or preprocessing steps will automatically generate synthetic data to improve model generalization.

---

## 🖥️ How to Run

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd llm
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Steps in the App**

   * Enter your Gemini API key.
   * Upload an Excel file.
   * Type an analysis prompt (e.g., “Predict sales trends”).
   * Choose one of the suggested algorithms.
   * View visual insights and export the Word report.

---

## 📚 Example Prompts

* “Predict employee attrition based on HR dataset.”
* “Cluster customer purchase behavior.”
* “Forecast sales for the next 6 months.”
* “Identify key features influencing profit margin.”

---

## 🧾 Output

* Suggested Algorithms
* Executed ML Code (Decision Tree, Random Forest, Prophet, etc.)
* Metrics (Accuracy, RMSE, etc.)
* Data Visualizations
* Downloadable Word Report (`AI_Excel_Report.docx`)

---

## 🧑‍💻 Author

**Nagadevara Veera Subrahmanyam , Kaduru Sujitha**
Project: *AI Excel Analyzer (Gemini)*

---

**✨ Smart Data Analysis. Powered by Gemini.**
