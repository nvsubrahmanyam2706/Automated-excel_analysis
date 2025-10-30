# dashboard_agent.py
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

def generate_dashboard(df: pd.DataFrame, prompt: str):
    """
    Generate a KPI dashboard (with visualizations) for the uploaded dataset.
    This function does not use Gemini fallbacks — it adapts automatically to any dataset.
    """

    dashboard_images = []
    try:
        # Basic dataset KPIs
        num_rows, num_cols = df.shape
        missing_vals = int(df.isnull().sum().sum())

        numeric_df = df.select_dtypes(include=np.number)
        avg_numeric = numeric_df.mean().mean() if not numeric_df.empty else 0

        kpi_text = f"""🧾 **Key Performance Indicators (KPIs)**

- **Rows:** {num_rows}
- **Columns:** {num_cols}
- **Missing Values:** {missing_vals}
- **Average of Numeric Columns:** {round(avg_numeric, 2)}
"""

        # -------------- Generate Visuals (Grid Layout Ready) -----------------
        # Create 4 adaptable visualizations depending on dataset structure
        visual_plots = []

        # Histogram for numeric columns
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            numeric_df.hist(ax=ax)
            plt.tight_layout()
            visual_plots.append(fig)

        # Correlation heatmap
        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True, ax=ax)
            ax.set_title("Correlation Heatmap")
            visual_plots.append(fig)

        # Top category counts (for first categorical column)
        cat_cols = df.select_dtypes(exclude=np.number).columns
        if len(cat_cols) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            df[cat_cols[0]].value_counts().head(10).plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title(f"Top Categories ({cat_cols[0]})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            visual_plots.append(fig)

        # Numeric trend for first numeric column
        if not numeric_df.empty:
            first_num = numeric_df.columns[0]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df[first_num], marker='o', color='orange')
            ax.set_title(f"Trend of {first_num}")
            plt.tight_layout()
            visual_plots.append(fig)

        # Convert all matplotlib figs to bytes
        for fig in visual_plots:
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            dashboard_images.append(buf.read())
            plt.close(fig)

        return kpi_text, dashboard_images

    except Exception as e:
        # Log in Streamlit UI, not fallback
        return f"⚠️ KPI Dashboard generation error: {str(e)}", []
