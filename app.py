# app.py
import streamlit as st
import base64
import tempfile
import datetime
import qdrant_client

from utils import (
    load_pdf, chunk_text, embed_and_store,
    retrieve_context, ask_llm, speak_text
)

# Constants
COLLECTION_NAME = "pdf_chunks"

# --- Session State Initialization ---
for key, default in {
    "pdf_processed": False,
    "vectorstore": None,
    "chat_history": [],
    "last_question": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Page Setup ---
st.set_page_config(page_title="Notebook ChatMuse", layout="wide")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Custom styling
add_bg_from_local("download.png")
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
    .chat-question {
        background-color: #1d4ed8; color: white; font-weight: 500;
        padding: 12px 16px; border-radius: 18px;
        margin-bottom: 6px; max-width: 80%; align-self: flex-end;
    }
    .chat-answer {
        background-color: #475569; color: #e2e8f0;
        padding: 12px 16px; border-radius: 18px;
        margin-bottom: 16px; max-width: 80%; align-self: flex-start;
    }
    .stButton>button {
        background-color: #c4b5fd; color: black;
        border: none; border-radius: 10px;
        padding: 10px 20px; font-size: 16px;
    }
    .stButton>button:hover { background-color: #059669; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("📄 Notebook ChatMuse")
st.markdown("Upload a PDF and ask questions based on its content.")

# --- Upload PDF ---
pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if pdf_file and not st.session_state.pdf_processed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    # Clear previous collection
    client = qdrant_client.QdrantClient("http://localhost:6333")
    try:
        client.delete_collection(COLLECTION_NAME)
        st.info(f"Old Qdrant collection '{COLLECTION_NAME}' cleared.")
    except Exception:
        pass

    with st.spinner("🔍 Reading & chunking PDF..."):
        pages = load_pdf(tmp_path)
        chunks = chunk_text(pages)

    with st.spinner("🔗 Embedding & storing in Qdrant..."):
        vectorstore = embed_and_store(chunks, COLLECTION_NAME)

    st.session_state.vectorstore = vectorstore
    st.session_state.pdf_processed = True
    st.success("✅ PDF processed and stored successfully!")

# --- Chat Interface ---
if st.session_state.pdf_processed and st.session_state.vectorstore:
    st.subheader("💬 Ask a question")

    with st.form("question_form", clear_on_submit=True):
        query = st.text_input("Type your question:")
        submitted = st.form_submit_button("Ask ChatMuse")

    if submitted and query and query != st.session_state.last_question:
        with st.spinner("🤖 Thinking..."):
            retriever = retrieve_context(query, st.session_state.vectorstore)
            answer = ask_llm(query, retriever)

        st.session_state.chat_history.append((query, answer))
        st.session_state.last_question = query

# --- Display Chat History ---
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 🧠 Your Question with answer")

    latest_q, latest_a = st.session_state.chat_history[-1]
    st.markdown(f'<div class="chat-question">🙋‍♂️ <b>You:</b> {latest_q}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-answer">🤖 <b>Bot:</b> {latest_a}</div>', unsafe_allow_html=True)

    audio_data_url = speak_text(latest_a)
    st.audio(audio_data_url, format="audio/mp3")

# --- Display Older Questions ---
if len(st.session_state.chat_history) > 1:
    st.markdown("---")
    st.markdown("### 🗂️ Previous Questions")

    for i, (q, a) in enumerate(reversed(st.session_state.chat_history[:-1]), 1):
        q_number = len(st.session_state.chat_history) - i
        st.markdown(f'<div class="chat-question">Q{q_number}: {q}</div>', unsafe_allow_html=True)
        with st.expander("🔽 Show Answer", expanded=False):
            st.markdown(f'<div class="chat-answer">🤖 <b>Bot:</b> {a}</div>', unsafe_allow_html=True)

# --- Save Chat History ---
if st.session_state.chat_history:
    if st.button("💾 Save Chat History"):
        chat_text = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        file_name = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button("📥 Download Chat", chat_text, file_name=file_name, mime="text/plain")

# --- Extra Action Buttons ---
if st.session_state.chat_history:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.last_question = ""
            st.rerun()
    with col2:
        if st.button("📤 Upload Another PDF"):
            for key in ["pdf_processed", "vectorstore", "chat_history", "last_question"]:
                st.session_state[key] = False if key == "pdf_processed" else None if key == "vectorstore" else []
            st.rerun()
