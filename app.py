import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
import io
import json

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

    st.markdown("### 💬 Ask a question about your data to generate a chart")

    question = st.text_input("Example: What companies were there? OR Show distribution of age by department")

    if question:
        with st.spinner("🔍 Interpreting your question..."):
            # Build smart prompt
            prompt = f"""
You are a data visualization assistant.

Given this dataset sample:
{df.head(10).to_csv(index=False)}

And these column names: {', '.join(df.columns)}

Interpret the user's question and return a JSON object that specifies:
1. chart_type: one of ['bar', 'pie', 'hist', 'box', 'scatter']
2. x: column to plot on X-axis
3. y: column to plot on Y-axis (null if not needed)
4. operation: optional (e.g., 'count', 'mean', 'sum', or null)
5. filter: optional (e.g., "Gender == 'Female'" or null)

User's question: {question}

Respond only with a JSON object. No explanation.
            """

            try:
                response = model.generate_content(prompt)
                chart_config = json.loads(response.text.strip())

                st.markdown("### 🔧 Gemini's Visualization Plan")
                st.json(chart_config)

                chart_type = chart_config.get("chart_type")
                x = chart_config.get("x")
                y = chart_config.get("y")
                operation = chart_config.get("operation")
                filter_expr = chart_config.get("filter")

                # Apply filtering if specified
                if filter_expr:
                    try:
                        df_filtered = df.query(filter_expr)
                    except Exception:
                        st.warning("⚠️ Invalid filter expression. Showing full dataset.")
                        df_filtered = df
                else:
                    df_filtered = df

                st.markdown("### 📊 Generated Chart")
                plt.figure(figsize=(10, 5))

                # Generate chart based on intent
                if chart_type == "bar":
                    if operation == "count":
                        sns.countplot(data=df_filtered, x=x)
                    elif operation in ["mean", "sum"] and y:
                        grouped = df_filtered.groupby(x)[y].agg(operation).reset_index()
                        sns.barplot(data=grouped, x=x, y=y)
                    else:
                        sns.barplot(data=df_filtered, x=x, y=y)

                elif chart_type == "pie":
                    if operation == "count":
                        pie_data = df_filtered[x].value_counts()
                    else:
                        pie_data = df_filtered.groupby(x)[y].agg(operation)
                    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%')
                    plt.axis('equal')

                elif chart_type == "hist":
                    sns.histplot(data=df_filtered, x=x, kde=True)

                elif chart_type == "box":
                    sns.boxplot(data=df_filtered, x=x, y=y)

                elif chart_type == "scatter":
                    sns.scatterplot(data=df_filtered, x=x, y=y)

                else:
                    st.warning("🤖 Model returned unknown chart type.")

                st.pyplot(plt.gcf())
                plt.clf()

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")