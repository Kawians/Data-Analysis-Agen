import streamlit as st
import pandas as pd
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel("gemini-pro")

# Streamlit App
st.title("🧠 AI Data Analysis Assistant")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data")

    if question:
        prompt = f"""You're a data analyst. Here's a preview of the dataset:
        {df.head().to_csv(index=False)}
        
        Column names: {', '.join(df.columns)}
        
        Now answer this question: {question}
        """
        response = model.generate_content(prompt)
        st.markdown("### 📊 Answer")
        st.write(response.text)