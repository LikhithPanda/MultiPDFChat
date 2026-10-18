"""
MultiPDFChat - Streamlit Web App
Chat with multiple PDF files using Google Gemini + LangChain + FAISS.

Run locally:
    streamlit run app.py
"""

import os
import tempfile
import asyncio

import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai


try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


EMBEDDING_MODEL = "models/gemini-embedding-001"
# If your API key doesn't have access to that model, switch to the older,
# broadly-available one instead:
# EMBEDDING_MODEL = "models/embedding-001"


# PAGE CONFIG
st.set_page_config(page_title="MultiPDFChat", page_icon="📄", layout="wide")


# API KEY HANDLING
def get_api_key():
    """
    Looks for the Gemini API key in this order:
    1. st.secrets["GOOGLE_API_KEY"]   (used on Streamlit Community Cloud)
    2. the GOOGLE_API_KEY environment variable (used for local/Docker runs)
    3. a text box in the sidebar, as a last resort so the app never hardcodes a key
    """
    key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
    if not key:
        key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        key = st.session_state.get("manual_api_key")
    return key

def ensure_api_key_configured():
    key = get_api_key()
    if not key:
        st.sidebar.warning("No Gemini API key found in secrets or environment.")
        manual_key = st.sidebar.text_input(
            "Enter your Gemini API key", type="password", key="manual_api_key"
        )
        if manual_key:
            key = manual_key
            st.rerun()
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        genai.configure(api_key=key)
    return key


# PDF -> text -> chunks -> FAISS (same logic as the notebook, no Colab/Drive)
def get_pdf_text(pdf_files):
    """pdf_files: list of file-like objects from st.file_uploader."""
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def build_vector_store(text_chunks):
    """
    Builds an in-memory FAISS index and returns it directly, instead of
    saving to / loading from disk. This is what lets the app work with
    whatever PDFs the current user uploaded, without needing persistent
    storage or Google Drive.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt = ChatPromptTemplate.from_template("""
    Answer the question as detailed as possible using ONLY
    the information provided in the context.

    If the answer is not available in the context, say:
    "Answer is not available in the context."

    Do not use outside knowledge
    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    return prompt | model | StrOutputParser()

def answer_question(vector_store, user_question):
    docs = vector_store.similarity_search(user_question, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = get_conversational_chain()
    return chain.invoke({"context": context, "question": user_question})


# STREAMLIT UI
def main():
    st.title("Chat with Multiple PDFs (Gemini + RAG)")
    st.caption("Upload PDFs, build an index, then ask questions grounded only in their content.")

    api_key = ensure_api_key_configured()

    # Session state holds the FAISS index and chat history so they survive
    # reruns (Streamlit reruns the whole script on every interaction).
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (question, answer) tuples
    if "indexed_file_names" not in st.session_state:
        st.session_state.indexed_file_names = []

    with st.sidebar:
        st.header("1. Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose one or more PDF files", type="pdf", accept_multiple_files=True
        )

        process_clicked = st.button("Process PDFs", type="primary", disabled=not api_key)
        if not api_key:
            st.info("Enter your Gemini API key above to enable processing.")

        if process_clicked:
            if not uploaded_files:
                st.error("Please upload at least one PDF first.")
            else:
                with st.spinner("Extracting text, splitting into chunks, and building the index..."):
                    raw_text = get_pdf_text(uploaded_files)
                    if not raw_text.strip():
                        st.error("Couldn't extract any text from the uploaded PDF(s). "
                                 "They may be scanned images without a text layer.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        st.session_state.vector_store = build_vector_store(text_chunks)
                        st.session_state.indexed_file_names = [f.name for f in uploaded_files]
                        st.session_state.chat_history = []
                        st.success(
                            f"Indexed {len(uploaded_files)} file(s), "
                            f"{len(raw_text):,} characters, {len(text_chunks)} chunks."
                        )

        if st.session_state.indexed_file_names:
            st.markdown("**Currently indexed:**")
            for name in st.session_state.indexed_file_names:
                st.markdown(f"- {name}")

    st.header("2. Ask a question")

    if st.session_state.vector_store is None:
        st.info("Upload PDF(s) and click **Process PDFs** in the sidebar to get started.")
        return

    user_question = st.chat_input("Ask a question about your PDFs...")

    if user_question:
        with st.spinner("Searching your documents and asking Gemini..."):
            answer = answer_question(st.session_state.vector_store, user_question)
        st.session_state.chat_history.append((user_question, answer))

    # Render chat history, most recent last (standard chat order)
    for question, answer in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)

if __name__ == "__main__":
    main()