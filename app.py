import os
import re
import tempfile
from typing import Optional

import pymupdf
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return default


HF_TOKEN = get_setting("HF_TOKEN")
GROQ_API_KEY = get_setting("GROQ_API_KEY")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

st.set_page_config(
    page_title="AI Agent with RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_text" not in st.session_state:
    st.session_state.knowledge_text = ""
if "knowledge_source" not in st.session_state:
    st.session_state.knowledge_source = ""


def extract_pdf_text(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    doc = pymupdf.open(temp_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    os.remove(temp_path)
    return "\n\n".join(page for page in pages if page).strip()


def build_knowledge_base(uploaded_file) -> Optional[str]:
    with st.spinner("📥 Processing PDF..."):
        try:
            text = extract_pdf_text(uploaded_file)
        except Exception as exc:
            st.error(f"PDF processing failed: {exc}")
            return None

        if not text:
            st.error("The uploaded PDF did not contain readable text.")
            return None

        st.session_state.knowledge_text = text
        st.session_state.knowledge_source = uploaded_file.name
        return text


def safe_calculate(expression: str) -> str:
    expression = expression.strip()
    if not re.fullmatch(r"[0-9\s+\-*/().]+", expression):
        return "Only basic arithmetic is supported."
    try:
        allowed_names = {"abs": abs, "min": min, "max": max, "round": round}
        return str(eval(expression, {"__builtins__": None}, allowed_names))
    except Exception as exc:
        return f"Calculation error: {exc}"


def answer_question(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a question or command."

    if prompt.lower().startswith("calc"):
        expr = prompt[4:].strip()
        return safe_calculate(expr)

    if re.fullmatch(r"[0-9\s+\-*/().]+", prompt):
        return safe_calculate(prompt)

    if prompt.lower().startswith("summarize") and st.session_state.knowledge_text:
        return st.session_state.knowledge_text[:1200]

    if st.session_state.knowledge_text:
        needle = prompt.lower()
        text = st.session_state.knowledge_text.lower()
        if needle in text:
            idx = text.find(needle)
            start = max(0, idx - 180)
            end = min(len(st.session_state.knowledge_text), idx + 500)
            return st.session_state.knowledge_text[start:end] + "..."

        return (
            "I can search the uploaded PDF content. Try asking a question such as 'what is in this document?'"
        )

    return (
        "The app is ready. Upload a PDF in the sidebar to enable local document search. "
        "You can also use simple arithmetic such as 'calc 2 + 2'."
    )


with st.sidebar:
    st.header("⚙️ Codespace Setup")
    st.write("This workspace is configured to run directly in the Codespace with Streamlit.")
    st.divider()
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.button("Build knowledge base", use_container_width=True):
        if uploaded_pdf is not None:
            build_knowledge_base(uploaded_pdf)
            st.success("Knowledge base ready.")
        else:
            st.warning("Please upload a PDF first.")

    st.divider()
    st.caption("Status: local startup path enabled")


st.title("🤖 AI Agent with RAG")
st.markdown("This version runs directly in the Codespace and supports PDF search plus simple calculator actions.")

if st.session_state.knowledge_source:
    st.info(f"Loaded document: {st.session_state.knowledge_source}")
else:
    st.info("Upload a PDF to activate local document search.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the document or use a calculation"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    reply = answer_question(prompt)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
