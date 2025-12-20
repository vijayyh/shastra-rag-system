import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"


def load_documents():
    documents = []
    for file in os.listdir(DATA_DIR):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            documents.extend(loader.load())
    return documents


def main():
    documents = load_documents()

    if not documents:
        print("❌ No PDF files found inside data/ folder")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(VECTORSTORE_DIR)

    print("✅ Vector database created successfully")


if __name__ == "__main__":
    main()
