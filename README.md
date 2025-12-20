# Shastra RAG LLM

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on the Bhagavad Gita (or other PDF documents) using Groq's Llama 3.1 model and LangChain.

## Features

- **Document Ingestion**: Loads PDF documents from a `data/` directory.
- **Vector Store**: Uses FAISS and HuggingFace embeddings (`all-MiniLM-L6-v2`) for efficient similarity search.
- **LLM Integration**: Custom integration with Groq API using `llama-3.1-8b-instant`.
- **Interactive Chat**: Console-based interface for querying the document context.

## Prerequisites

- Python 3.8+
- A Groq API Key

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install langchain langchain-community faiss-cpu python-dotenv groq sentence-transformers pypdf
```

3. Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### 1. Ingest Data

Place your PDF files (e.g., Bhagavad Gita) inside the `data/` folder. Then run the ingestion script to create the vector database:

```bash
python ingest.py
```

This will generate a `vectorstore/` directory containing the FAISS index.

### 2. Run the Chatbot

Start the interactive chatbot:

```bash
python chatbot.py
```

Type your questions when prompted. Type `exit` to quit.