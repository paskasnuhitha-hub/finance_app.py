import streamlit as st
import requests
import os

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Finance AI with Granite", page_icon="💰")
st.title("💰 Finance AI Assistant (Granite via Hugging Face)")
st.write("This demo uses **IBM Granite** models from Hugging Face Inference API.")

# -------------------------------
# Hugging Face API Setup
# -------------------------------
# Directly using the Hugging Face Token (use secrets or env variable in production)
HF_TOKEN = "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"
API_URL_QA = "https://api-inference.huggingface.co/models/ibm-granite/granite-7b-instruct"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(payload, api_url=API_URL_QA):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs([
    "📘 QA: Ask Financial Questions",
    "📊 Sentiment on Financial News",
    "📰 Summarize Finance Text"
])

# -------------------------------
# 1. QA
# -------------------------------
with tab1:
    st.header("📘 Financial Question Answering (Granite)")
    context = st.text_area("Enter Financial Context (e.g., company reports, policy, investment info):")
    question = st.text_input("Ask a financial question related to the above:")

    if st.button("Answer", key="qa_btn"):
        if context and question:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer clearly:"
            with st.spinner("Granite is thinking..."):
                output = query({"inputs": prompt})
            st.success(output[0]["generated_text"] if isinstance(output, list) else str(output))
        else:
            st.warning("Please provide both context and a question.")

# -------------------------------
# 2. Sentiment
# -------------------------------
with tab2:
    st.header("📊 Sentiment Analysis with Granite")
    news = st.text_area("Paste a financial news headline or snippet:")

    if st.button("Analyze Sentiment", key="sent_btn"):
        if news:
            prompt = f"Classify the sentiment (Positive, Negative, Neutral) of this financial news:\n\n{news}"
            with st.spinner("Granite is analyzing..."):
                output = query({"inputs": prompt})
            st.success(output[0]["generated_text"] if isinstance(output, list) else str(output))
        else:
            st.warning("Please enter some news text.")

# -------------------------------
# 3. Summarization
# -------------------------------
with tab3:
    st.header("📰 Financial Text Summarizer with Granite")
    text = st.text_area("Enter a long financial article or report:")

    if st.button("Summarize", key="sum_btn"):
        if text:
            prompt = f"Summarize the following financial text in simple terms:\n\n{text}"
            with st.spinner("Granite is summarizing..."):
                output = query({"inputs": prompt})
            st.success(output[0]["generated_text"] if isinstance(output, list) else str(output))
        else:
            st.warning("Please enter the text to summarize.")
