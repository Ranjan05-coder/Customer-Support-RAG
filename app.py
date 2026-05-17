import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

# Setup - Use st.secrets for Streamlit Cloud, fallback to os.getenv for local
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="groq/compound", groq_api_key=groq_api_key)
hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
os.environ['HF_TOKEN'] = hf_token
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="Chat Customer Service", layout="wide")
st.title("📄 Chat with Customer Service")

# Load and process PDF
try:
    pdf_path = os.path.join(os.path.dirname(__file__), "src", "customer_support_guide.pdf")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    retriever = vectorstore.as_retriever()
except Exception as e:
    st.error(f"Error loading PDF: {e}")
    retriever = None

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    ai_greeting = "Hi, I'm a Customer Service bot who can help you about your problem. Don't be shy to ask!"
    st.session_state.messages.append({"role": "assistant", "content": ai_greeting})

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input and retriever:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant docs
    docs = retriever.invoke(user_input)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate response
    system_prompt = (
        "You are a helpful customer service assistant. "
        "Use the following context to answer questions. "
        "If you don't know, say so. Keep answers concise.\n\n"
        f"Context: {context}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    response = llm.invoke(messages)
    bot_response = response.content

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)
elif user_input and not retriever:
    st.error("PDF not loaded. Check the file path.")
