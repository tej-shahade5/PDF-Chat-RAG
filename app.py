import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, model_name="Google AI"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

@st.cache_resource(show_spinner="Initializing Embeddings...")
def get_embeddings(model_name="Google AI", api_key=None):
    if model_name == "Google AI" and api_key:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    return None

def get_vector_store(text_chunks, embeddings, persist_name="faiss_index"):
    if embeddings:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(persist_name)
        return vector_store
    return None

@st.cache_resource(show_spinner="Initializing Chat Model...")
def get_chat_model(model_name="Google AI", api_key=None):
    if model_name == "Google AI" and api_key:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    return None

def get_conversational_chain(llm):
    if llm:
        prompt_template = """Answer the question as detailed as possible from the provided context. Be comprehensive and provide all relevant information. If the answer is not explicitly found in the context, state clearly "The answer is not available in the provided context." Do not fabricate information.

Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
        return chain
    return None

def display_chat_message(question, answer, is_user=True):
    role = "user" if is_user else "bot"
    st.markdown(
        f"""
        <div class="chat-message {role}">
            <div class="message-content">
                <div class="message-meta">{question}</div>
                <div class="message-text">{answer}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if not pdf_docs or not api_key:
        st.warning("Please upload PDF files and provide API key before processing.")
        return

    with st.spinner("Processing your query..."):
        text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)
        embeddings = get_embeddings(model_name, api_key)
        vector_store = get_vector_store(text_chunks, embeddings)
        llm = get_chat_model(model_name, api_key)
        chain = get_conversational_chain(llm)

        if vector_store and chain:
            docs = vector_store.similarity_search(user_question)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            answer = response['output_text']
            pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
            conversation_history.append({
                "question": user_question,
                "answer": answer,
                "model": model_name,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "pdf_names": ", ".join(pdf_names)
            })

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with your PDFs :books:")

    # Initialize session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = None

    st.sidebar.markdown("""
        <div class="social-links">
            <a href="https://www.linkedin.com/in/tejas-shahade-956b30223/" target="_blank"><i class="fab fa-linkedin"></i> LinkedIn</a>
            <a href="https://github.com/tej-shahade5" target="_blank"><i class="fab fa-github" style="color: white"></i> GitHub</a>
        </div>
    """, unsafe_allow_html=True)

    model_name = st.sidebar.radio("Select the Model:", ("Google AI",))
    api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
    st.sidebar.markdown("Click [here](https://ai.google.dev/) to get an API key.")

    with st.sidebar:
        st.title("Menu:")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset Chat", use_container_width=True)
        clear_button = col1.button("Clear Input", use_container_width=True)

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.pdf_docs = None

        st.session_state.pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        submit_button = st.button("Submit & Process", use_container_width=True)
        if submit_button and st.session_state.pdf_docs:
            with st.spinner("Processing PDF(s)..."):
                get_pdf_text(st.session_state.pdf_docs)
                st.success("PDF(s) processed!")
        elif submit_button and not st.session_state.pdf_docs:
            st.warning("Please upload PDF files before processing.")

        if st.session_state.conversation_history:
            df = pd.DataFrame(st.session_state.conversation_history)
            csv = df.to_csv(index=False).encode()
            b64 = base64.b64encode(csv).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button class="download-btn">Download History</button></a>'
            st.sidebar.markdown(href, unsafe_allow_html=True)

    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input("Ask a Question from the PDF Files")
        submit_question = st.form_submit_button(label="Submit Question")

    if submit_question and user_question and st.session_state.get("pdf_docs") and api_key:
        user_input(user_question, model_name, api_key, st.session_state.pdf_docs, st.session_state.conversation_history)

    for chat in reversed(st.session_state.conversation_history):
        display_chat_message(chat["question"], chat["answer"], is_user=True)
        # display_chat_message(chat["question"], chat["answer"], is_user=False)

    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <style>
        .chat-message {
            margin: 15px 0;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .chat-message.user {
            background: linear-gradient(135deg, #667eea, #764ba2);
            margin-left: 20%;
            color: white;
        }
        .chat-message.bot {
            background: #ffffff;
            margin-right: 20%;
            border: 1px solid #eee;
        }
        .message-content {
            position: relative;
        }
        .message-text {
            font-size: 16px;
            line-height: 1.5;
        }
        .message-meta {
            font-size: 12px;
            opacity: 0.7;
            margin-top: 5px;
        }
        .chat-message.user .message-meta {
            color: #ddd;
        }
        .social-links {
            display: flex;
            justify-content: space-between;
        }
        .social-links a {
            display: block;
            margin: 15px 0;
            color: gray;
            text-decoration: none;
            font-size: 16px;
            transition: color 0.3s ease;
        }
        .social-links a:hover {
            color: white;
        }
        .social-links i {
            font-size: 24px; /* Larger icons */
            margin-right: 10px;
            vertical-align: middle;
            color: white;
        }
        .social-links a:nth-child(2) i {
            color: #24292e; /* Darker GitHub gray */
        }
        .download-btn {
            width: 100%;
            padding: 8px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .download-btn:hover {
            background: #45a049;
        }
        .st-emotion-cache-kgpedg {
            padding: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
