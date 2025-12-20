📜 Bhagavad Gita RAG System

Retrieval-Augmented Generation using LangChain, FAISS, and Groq

📌 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers user queries strictly based on the Bhagavad Gita PDF.

Instead of relying on a general-purpose language model, the system first retrieves relevant content from the document and then generates answers grounded in that retrieved context. This avoids hallucinations and ensures document-faithful responses.

🎯 Objective

The objective of this project is to:

Build a document-grounded chatbot

Demonstrate a complete RAG pipeline

Use embeddings and vector search for retrieval

Integrate a Large Language Model (LLM) for answer generation

This project was implemented as part of the Shastra_AI RAG system task.

🧠 What is Retrieval-Augmented Generation (RAG)?

RAG combines two core ideas:

Retrieval
Relevant document chunks are retrieved from a vector database using semantic similarity.

Generation
A language model generates an answer using both the user query and the retrieved context.

This ensures:

Answers come from the document

Reduced hallucinations

Explainable AI behavior

🏗️ System Architecture
Bhagavad Gita PDF
        ↓
Text Chunking
        ↓
Embeddings (Sentence Transformers)
        ↓
FAISS Vector Store
        ↓
Retriever
        ↓
Groq LLM
        ↓
Final Answer

🛠️ Technologies Used
Programming Language

Python 3.11

Libraries & Tools

LangChain – RAG pipeline and chains

FAISS – Vector similarity search

Sentence Transformers – Text embeddings

Groq SDK – Large Language Model inference

python-dotenv – Environment variable management

PyPDF – PDF loading

📦 Project Structure
shastra_rag_llm/
│
├── data/
│   └── bhagavad_gita.pdf
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── ingest.py
├── chatbot.py
├── requirements.txt
├── README.md
└── .env   (not included in submission)

⚙️ How to Run This Project (Exact Steps)
🔹 Prerequisites

Python 3.11.x

Internet connection

Groq API Key

Check Python version:

python --version

🔹 Step 1: Create Virtual Environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


You should see:

(venv)

🔹 Step 2: Install Dependencies

All dependencies are listed in requirements.txt.

pip install -r requirements.txt


This installs LangChain, FAISS, Groq SDK, and all required libraries.

🔹 Step 3: Set Up Environment Variables

Create a .env file in the project root directory:

GROQ_API_KEY=your_groq_api_key_here


⚠️ Do not share this file publicly.

🔹 Step 4: Run Document Ingestion (One-Time)

This step processes the Bhagavad Gita PDF and creates the vector database.

python ingest.py


After successful execution, a vectorstore/ folder will be created.

🔹 Step 5: Run the Chatbot

Start the RAG chatbot:

python chatbot.py


Expected output:

📜 Bhagavad Gita RAG Chatbot Ready
Type 'exit' to quit

🔹 Step 6: Ask Questions

Example questions:

What is Karma Yoga according to the Bhagavad Gita?

What advice does Krishna give to Arjuna?

Explain Nishkama Karma

What is Dharma in the Gita?

Type exit to stop the chatbot.

🧪 Validation (Proving It Is RAG)

Ask an unrelated question:

Who is the Prime Minister of India?


The system should not answer correctly, proving that responses are limited to the provided document.

📚 Concepts Used in This Project

Retrieval-Augmented Generation (RAG)

Vector embeddings

Cosine similarity

FAISS indexing

Prompt grounding

LLM API integration

Environment variable security

🧹 Notes for Submission

venv/ folder is not included

.env file is not shared

Dependencies are reproducible using requirements.txt

🚀 Future Enhancements

Web interface (Streamlit / React)

Support for multiple PDFs

Answer citations

Cloud deployment

User authentication

🧾 Conclusion

This project demonstrates a complete and practical implementation of a Retrieval-Augmented Generation system. It shows how modern LLMs can be safely combined with document retrieval to produce accurate, grounded, and explainable responses.