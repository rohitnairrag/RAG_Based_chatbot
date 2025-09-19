import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextractor import text_extractor_pdf

# -------------------- Streamlit Config --------------------
st.set_page_config(
    page_title="AI-Powered RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------- Sidebar --------------------
st.sidebar.title("📂 Upload your Document")
file_uploader = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.info("💡 Upload a PDF and start chatting with it instantly!")

# -------------------- Initialize --------------------
file_text = ""
if file_uploader is not None:
    file_text = text_extractor_pdf(file_uploader)

# -------------------- Main Title --------------------
st.title("🤖 AI-Powered RAG Chatbot")
st.markdown(
    """
    <div style="background-color:#f0f8ff;padding:15px;border-radius:10px;">
    <b>How it works:</b>  
    1️⃣ Upload a PDF document using the sidebar.  
    2️⃣ Ask any question about the document.  
    3️⃣ Get context-aware answers powered by RAG + Gemini AI.  
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Step 1 : Configure Models --------------------
key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)
llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------- Step 2 : Split into Chunks --------------------
chunks = []
if file_text:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

# -------------------- Step 3 : Create FAISS Vector Store --------------------
vector_store = None
if chunks:
    vector_store = FAISS.from_texts(chunks, embedding_model)

# -------------------- Step 4 : Configure Retriever --------------------
retriever = None
if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -------------------- Chat Memory --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------- Chat Input --------------------
query = st.chat_input("💬 Ask me anything about your document...")

if query:
    if query.lower() in ['exit', 'quit', 'stop', 'bye', 'goodbye', 'close', 'end']:
        st.chat_message("assistant").write("👋 Chat ended. Have a great day!")
        st.stop()

    elif not file_text:
        st.error("⚠️ Please upload a PDF before asking questions.")

    else:
        # Retrieve context
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build prompt
        prompt = f"""
        You are a highly professional AI assistant. 
        Use the provided context from the document to answer the question clearly and concisely. 
        If the context does not contain the answer, politely say you don’t know.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        # Generate response
        response = llm_model.generate_content(prompt)

        # Save to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": response.text})

# -------------------- Display Chat History --------------------
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
