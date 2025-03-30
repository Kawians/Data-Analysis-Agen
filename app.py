# import streamlit as st
# import pandas as pd
# import google.generativeai as genai

# # Configure Gemini
# import os
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

# # Streamlit App
# st.title("🧠 AI Data Analysis Assistant")

# uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

# if uploaded_file:
#     # Read file
#     if uploaded_file.name.endswith(".csv"):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)

#     st.dataframe(df.head())

#     question = st.text_input("Ask a question about your data")

#     if question:
#         prompt = f"""You're a data analyst. Here's a preview of the dataset:
#         {df.head().to_csv(index=False)}
        
#         Column names: {', '.join(df.columns)}
        
#         Now answer this question: {question}
#         """
#         response = model.generate_content(prompt)
#         st.markdown("### 📊 Answer")
#         st.write(response.text)


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
import io

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

    st.markdown("### 💬 Ask a question about your data or request a visualization")
    question = st.text_input("Example: What is the average age by department? OR Create a bar plot of sales by region.")

    if question:
        with st.spinner("🔍 Generating response..."):
            prompt = f"""
You are a Python data analyst. Based on this dataset:
{df.head().to_csv(index=False)}

Columns: {', '.join(df.columns)}

The user asked: {question}

Please write a Python function using seaborn and matplotlib to answer the question or generate the plot.
Return only the code that:
1. Uses seaborn/matplotlib
2. Uses 'df' as the dataset
3. Creates and shows the plot
4. Does NOT return text explanations
            """

            try:
                response = model.generate_content(prompt)
                code = response.text.strip("```python").strip("```")

                st.markdown("### 📊 Visualization")
                exec_globals = {'df': df, 'sns': sns, 'plt': plt, 'pd': pd}
                exec(code, exec_globals)
                st.pyplot(plt.gcf())
                plt.clf()

                st.markdown("### 🧠 Generated Code (for transparency)")
                st.code(code, language="python")

            except Exception as e:
                st.error(f"❌ Error running generated code: {e}")