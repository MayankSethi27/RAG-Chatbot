#  Conversational RAG Chatbot with PDF + Chat History

This project is a Conversational Retrieval-Augmented Generation (RAG) Chatbot built with **LangChain**, **Streamlit**, and **Groq LLM**. It allows users to upload a PDF file, ask questions, and receive answers using the document content — while maintaining session-based chat history.

---
##  Live App

 [Click here to try the RAG-Chatbot](https://rag-chatbot-mayank.streamlit.app/)

---
##  Features

-  Upload any PDF and extract its contents
-  Ask context-aware questions from the document
-  Maintains session-based chat history
-  Uses HuggingFace embeddings + ChromaDB 
-  Powered by Groq's Gemma-2 LLM via LangChain

---

##  Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Gemma-2)
- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Store**: Chroma with FAISS
- **LangChain Modules**: RAG chain, History-aware retriever, Document loaders

---

##  Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/MayankSethi27/RAG-Chatbot.git
cd RAG-Chatbot
```
### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 4. Set your environment variables

```bash
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_PROJECT=your_langchain_project_name
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
```
### 5. Run the app
```bash
streamlit run Chatbot.py
```
---
## Project Structure
```bash
RAG-Chatbot/
│
├── Chatbot.py               # Main Streamlit app
├── .env                     # Environment variables (ignored by Git)
├── .gitignore
├── requirements.txt
├── .devcontainer/           # Optional devcontainer configs
└── README.md
```
---
## Author
Mayank Sethi
