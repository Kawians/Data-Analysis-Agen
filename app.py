import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
import json
import re

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

st.set_page_config(page_title="AI Data Analysis Assistant", layout="wide")
st.title("🧠 AI Data Analysis Assistant")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Preview of Data")
    st.dataframe(df.head())

    st.markdown("### 💬 Ask a question about your data to get insights and visualizations")
    question = st.text_input("Example: What companies are in the dataset? OR Show average salary by department")

    if question:
        # -------- 1. Written Answer --------
        with st.spinner("🧠 Generating written insight..."):
            insight_prompt = f"""
You are a helpful data analyst.

Given this dataset:
{df.head(10).to_csv(index=False)}

Columns: {', '.join(df.columns)}

The user asked: {question}

Provide a short, clear written summary answering their question. It shouldn't be regarding how you are going to generate the visualization or calculate something.
You MUST answer their question based on the data directly. Be accurate and helpful.
            """
            try:
                insight_response = model.generate_content(insight_prompt)
                st.markdown("### ✍️ Insight")
                st.write(insight_response.text.strip())
            except Exception as e:
                st.error(f"❌ Failed to generate written answer: {e}")

        # -------- 2. Chart Planning --------
        with st.spinner("📊 Planning the visualization..."):
            schema_hint = json.dumps([
                {
                    "name": col,
                    "sample_values": df[col].dropna().astype(str).unique()[:3].tolist()
                }
                for col in df.columns
            ], indent=2)

            vis_prompt = f"""
You are a smart data visualization assistant.

Given this dataset sample:
{df.head(10).to_csv(index=False)}

Here is the column schema (column names + sample values):
{schema_hint}

Interpret the user's question and return a JSON object with:
- chart_type: one of ['bar', 'pie', 'hist', 'box', 'scatter']
- x: column name for X-axis
- y: column name for Y-axis (or null)
- operation: one of ['count', 'mean', 'sum', null]
- filter: optional string like "Gender == 'Female'" or null

User's question: {question}

Respond only with valid JSON. No explanation.
            """

            try:
                response = model.generate_content(vis_prompt)
                raw = response.text.strip()
                cleaned = re.sub(r"^```json|```$", "", raw, flags=re.IGNORECASE).strip()
                chart_config = json.loads(cleaned)

                # st.markdown("### 🔧 Gemini's Visualization Plan")
                # st.json(chart_config)

                chart_type = chart_config.get("chart_type")
                x = chart_config.get("x")
                y = chart_config.get("y")
                operation = chart_config.get("operation")
                filter_expr = chart_config.get("filter")

                # -------- 3. Filter Data --------
                if filter_expr:
                    try:
                        df_filtered = df.query(filter_expr)
                    except Exception:
                        st.warning("⚠️ Invalid filter expression. Showing full dataset.")
                        df_filtered = df
                else:
                    df_filtered = df

                # -------- 4. Generate Visualization --------
                st.markdown("### 📊 Generated Chart")
                plt.figure(figsize=(12, 6))
                TOP_N = 10

                if chart_type == "bar":
                    if operation == "count":
                        value_counts = df_filtered[x].value_counts().nlargest(TOP_N)
                        sns.barplot(x=value_counts.index, y=value_counts.values)
                        plt.ylabel("Count")
                        plt.xticks(rotation=45, ha='right')
                    elif operation in ["mean", "sum"] and y:
                        grouped = df_filtered.groupby(x)[y].agg(operation).nlargest(TOP_N).reset_index()
                        sns.barplot(data=grouped, x=x, y=y)
                        plt.xticks(rotation=45, ha='right')
                    else:
                        sns.barplot(data=df_filtered, x=x, y=y)
                        plt.xticks(rotation=45, ha='right')

                elif chart_type == "pie":
                    if operation == "count":
                        pie_data = df_filtered[x].value_counts().nlargest(TOP_N)
                    else:
                        pie_data = df_filtered.groupby(x)[y].agg(operation).nlargest(TOP_N)
                    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', textprops={'fontsize': 8})
                    plt.axis('equal')

                elif chart_type == "hist":
                    sns.histplot(data=df_filtered, x=x, kde=True)

                elif chart_type == "box":
                    sns.boxplot(data=df_filtered, x=x, y=y)
                    plt.xticks(rotation=45, ha='right')

                elif chart_type == "scatter":
                    sns.scatterplot(data=df_filtered, x=x, y=y)

                else:
                    st.warning("🤖 Gemini returned an unknown chart type.")

                st.pyplot(plt.gcf())
                plt.clf()

            except json.JSONDecodeError:
                st.error("❌ Gemini did not return valid JSON.")
            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")