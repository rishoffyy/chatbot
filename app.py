import streamlit as st
import tempfile
import os
from utils import process_pdf, create_vectorstore, get_answer

st.set_page_config(page_title="📄 AI PDF Chatbot", layout="wide")
st.title("📄 AI PDF Chatbot")
st.caption("Ask questions based on the content of your uploaded PDFs.")

uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
question = st.text_input("Ask a question based on the PDFs:")

if uploaded_files:
    with st.spinner("Processing..."):
        texts = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            texts.extend(process_pdf(tmp_path))
            os.remove(tmp_path)
        vectorstore = create_vectorstore(texts)

    if question:
        with st.spinner("Getting answer..."):
            answer = get_answer(vectorstore, question)
            st.markdown("### 💬 Answer")
            st.write(answer)
