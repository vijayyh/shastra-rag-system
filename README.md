# 🕉️ ShastraBot: AI Assistant for Vedic Scriptures

ShastraBot is a Retrieval-Augmented Generation (RAG) system designed to be a knowledgeable and versatile assistant for exploring Vedic scriptures. It provides factual, context-aware answers drawn from a local knowledge base and can be interacted with via a command-line interface, a web UI, a REST API, or by voice.

## ✨ Features

- **Hybrid Search:** Combines semantic (FAISS) and keyword-based (BM25) search for robust and accurate information retrieval.
- **Dynamic Query Expansion:** Automatically broadens short or ambiguous questions to find the best possible context.
- **Multiple Personas:** Adapts its response style based on the user's query, acting as a direct Q&A bot, a "teacher" for step-by-step explanations, or a "mind-mapper" for structured outlines.
- **Multiple Interfaces:**
  - **CLI:** A simple and direct command-line chat.
  - **Gradio Web UI:** An easy-to-use graphical interface for chatting in your browser.
  - **FastAPI:** A REST API for programmatic integration.
  - **Voice Chat:** Speak to the chatbot and hear its responses.
- **Source Citation:** Always lists the source documents it used to generate an answer.

## ⚙️ Architecture

The system works in two main phases:

1.  **Ingestion (`ingest.py`):**
    - Reads documents (`.pdf`, `.csv`, `.xlsx`) from the `data/` directory.
    - Splits the text into manageable chunks.
    - Uses the `BAAI/bge-base-en-v1.5` model to convert chunks into vector embeddings.
    - Stores these embeddings in a `FAISS` vector store located in the `vectorstore/` directory.

2.  **Inference (`chatbot.py` and interfaces):**
    - Takes a user query from any interface.
    - Sanitizes and normalizes the query.
    - Performs a hybrid search in the FAISS vector store to retrieve relevant context.
    - Selects a prompt persona based on the query.
    - Sends the query and context to the `llama-3.1-8b-instant` model via the Groq API.
    - Returns the generated answer and its sources to the user.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys)

### 2. Setup

**Clone the repository:**
```bash
git clone <your-repo-url>
cd shastra_rag_llm
```

**Create a virtual environment:**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# source venv/bin/activate    # On macOS/Linux
```

**Install dependencies:**
The `PyAudio` library can be difficult to install on Windows. A wheel file for Python 3.11 is included.
```bash
# First, try installing the PyAudio wheel
pip install PyAudio-0.2.11-cp311-cp311-win_amd64.whl

# Then, install the rest of the requirements
pip install -r requirements.txt
```

**Set up environment variables:**
Create a file named `.env` in the root of the project and add your Groq API key:
```
GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Data Ingestion
Before running the chatbot for the first time, you must build the vector store from the documents in the `data` folder.
```bash
python ingest.py
```
This will create the `vectorstore/` directory. You only need to run this again if you add or change documents in the `data/` directory.


## 🏃 How to Run ShastraBot

You can interact with the chatbot through any of the following interfaces.

### 1. Command-Line Interface (CLI)
For a simple, text-based chat in your terminal.
```bash
python chatbot.py
```

### 2. Gradio Web UI
To launch a user-friendly web interface.
```bash
python gradio_app.py
```
Open your web browser and navigate to the URL provided (usually `http://127.0.0.1:7860`).

### 3. FastAPI Server
To run the backend API. This is useful for integrating the chatbot into other applications.
```bash
uvicorn api:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can see the documentation at `http://127.0.0.1:8000/docs`.

### 4. Voice Chat
To have a spoken conversation with the chatbot.
```bash
python voice_chat.py
```

## 📂 Project Structure

```
├─── data/                # Source documents for the knowledge base
├─── vectorstore/         # FAISS vector store (created by ingest.py)
├─── .env                 # API keys and environment variables
├─── .gitignore
├─── api.py               # FastAPI application
├─── chatbot.log          # Log file for chatbot events
├─── chatbot.py           # Core RAG logic and CLI entry point
├─── config.json          # Configuration for models, retrieval, etc.
├─── gradio_app.py        # Gradio web UI application
├─── ingest.py            # Script for data ingestion and vector store creation
├─── README.md            # This file
├─── requirements.txt     # Python dependencies
├─── test_mic.py          # A simple script to test microphone input
├─── voice_chat.py        # Main entry point for the voice-enabled chat
├─── voice_input.py       # Handles microphone listening
└─── voice_output.py      # Handles text-to-speech
```
