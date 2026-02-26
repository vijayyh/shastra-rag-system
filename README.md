# 🕉️ ShastraBot: A Voice-Enabled RAG Chatbot

ShastraBot is a Retrieval-Augmented Generation (RAG) system designed to answer questions about Vedic scriptures and related texts. It can process documents, listen to voice commands, and provide answers through a command-line interface, a voice-based chat, or a web UI.

## ✨ Features

*   **Multi-Format Data Ingestion**: Ingests and processes data from `.pdf`, `.csv`, and `.xlsx` files.
*   **Vector-Based Retrieval**: Uses FAISS and sentence transformers for efficient and semantic document retrieval.
*   **Multiple Interfaces**: Interact via a standard text-based console, a hands-free voice chat, or a user-friendly Gradio web app.
*   **Dynamic Personas**: The chatbot can adopt different response styles (standard assistant, mind-map generator, or step-by-step teacher) based on the user's query.
*   **Intelligent Query Handling**: Includes query normalization, sanitization, and an automatic query expansion mechanism to improve retrieval accuracy for short or ambiguous questions.
*   **Robust Error Handling**: Features retry logic for LLM calls and graceful error recovery in the voice and chat interfaces.

## ⚙️ Setup and Installation

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd shastra-rag-system
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Important**: The `PyAudio` library, used for microphone input, has system-level dependencies. Please install them *before* running the pip command.

*   **On Windows**: `pip install pyaudio` usually works directly as it installs a pre-compiled wheel. [1] If it fails, you may need to install Microsoft C++ Build Tools. [7]
*   **On macOS**: `brew install portaudio` [1, 3]
*   **On Debian/Ubuntu**: `sudo apt-get install portaudio19-dev` [1, 2]

Once the `portaudio` dependency is met, install all the Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

The chatbot uses the Groq API for fast LLM inference. You need to provide an API key.

1.  Create a new file named `.env` in the root of the project directory.
2.  Add your API key to this file:

    ```
    GROQ_API_KEY="your_grok_api_key_here"
    ```

## 🚀 Usage

Make sure you have activated your virtual environment (`source venv/bin/activate`) before running any scripts.

### 1. Ingest Your Data

Place your source documents (`.pdf`, `.csv`, `.xlsx`) into the `data/` directory. Then, run the ingestion script to process them and create the vector store.

```bash
python ingest.py
```

### 2. Run the Chatbot

You can interact with ShastraBot in three different ways:

*   **Web Interface (Recommended)**:
    ```bash
    python gradio_app.py
    ```
    Open your browser and navigate to the local URL provided (e.g., `http://127.0.0.1:7860`).

*   **Voice Chat**:
    ```bash
    python voice_chat.py
    ```

*   **Text-based Console Chat**:
    ```bash
    python chatbot.py
    ```