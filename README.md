# RAG_Based PDF Chat-Assistant

The RAG-Based Chat Assistant is an intelligent PDF chatbot that leverages Retrieval-Augmented Generation (RAG) architecture to provide accurate and context-aware responses based on uploaded documents. By combining a vector database (FAISS) for semantic search and a powerful open-source LLM (LLaMA-3 70B) via Groq, the assistant retrieves relevant context and generates precise answers to user queries.

---

## Approach 

This assistant is built on the principle of retrieval-augmented generation, which enhances the reliability of large language models by grounding responses in document-specific information. The design follows a modular pipeline consisting of:

  1. Document Ingestion & Chunking: PDF documents are loaded and split into manageable chunks using RecursiveCharacterTextSplitter.

  2. Vector Store Creation: Chunks are embedded using sentence-transformers/all-mpnet-base-v2 and stored in a local FAISS vector database for fast retrieval.

  3. Multi-Query Retrieval: For a given user query, multiple diverse queries are generated to improve retrieval relevance using MultiQueryRetriever.

  4. Prompt Chaining & LLM Response: Retrieved context is formatted using a prompt template and passed to LLaMA-3 70B hosted via Groq for final response generation.

  5. Memory : Langgraph Memory is implemented to retain the last 3 human-AI conversations, enabling context-aware multi-turn conversations.

  6. Streamlit UI: An interactive and lightweight front-end allows users to chat with the assistant and view responses.

---
## Prerequisites

- Python 3.9 or higher
  
- [Create Groq API Key]( https://console.groq.com/keys)

## ⚙️ Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/prabal-k/Rag_based_PDF_Chatbot
```

### 2. Open with VsCode ,Create and Activate a Python Virtual Environment

### On Windows:
```
python -m venv venv

venv\Scripts\activate
```
### On Linux/macOS:
```
python3 -m venv venv

source venv/bin/activate
```
### 3. Install Required Dependencies
``
pip install -r requirements.txt
``
### 4. Configure Environment Variables

Create a .env file in the root folder with the following content:

GROQ_API_KEY = "your_groq_api_key_here"

### 5. Run the Application

```
streamlit run rag_chatbot_ui.py
``
