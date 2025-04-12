# PDF-Chat-RAG

A web-based conversational AI application built with Streamlit, LangChain, and Google Generative AI to query and extract insights from PDF documents using Retrieval-Augmented Generation (RAG).

![image](https://github.com/user-attachments/assets/4bbcafe4-04d3-4f34-b08f-5e1865607dd8)

## Overview

PDF-Chat-RAG enables users to upload multiple PDF files, ask questions in natural language, and receive detailed, context-aware responses. The application leverages a RAG pipeline to retrieve relevant document chunks and generate accurate answers using Google’s Gemini-2.0-flash model. This project demonstrates expertise in Generative AI, Natural Language Processing (NLP), vector embeddings.

### Key Features
- PDF Processing: Extracts text from uploaded PDFs using PyPDF2.
- RAG Pipeline: Combines FAISS vector store and Google Generative AI embeddings for efficient document retrieval and answer generation.
- Interactive UI: Built with Streamlit for a seamless user experience, including file uploads, question input, and conversation history.
- Conversation History: Stores and displays chat interactions with timestamps and PDF metadata using Pandas.

## Tech Stack
- Programming Language: Python
- Frameworks/Libraries: Streamlit, LangChain, PyPDF2, Pandas, FAISS
- AI/ML: Google Generative AI (Gemini-2.0-flash), Retrieval-Augmented Generation (RAG), NLP
- Other Tools: RecursiveCharacterTextSplitter, GoogleGenerativeAIEmbeddings

## Installation

To run the project locally, follow these steps:

1. Clone the Repository:
   git clone https://github.com/yourusername/PDF-Chat-RAG.git
   cd PDF-Chat-RAG
