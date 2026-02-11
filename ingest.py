import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_csv(file_path):
    docs = []

    encodings = ["utf-8", "utf-16", "latin-1", "ISO-8859-1"]

    df = None
    loaded_encoding = None
    
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, on_bad_lines="skip")
            loaded_encoding = enc
            print(f"✅ Loaded CSV with encoding: {enc} -> {file_path}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️ Error reading {file_path} with encoding {enc}: {e}")
            continue

    if df is None:
        print(f"❌ Failed to read CSV with any encoding: {file_path}")
        print(f"   Tried: {', '.join(encodings)}")
        print(f"   This file will be skipped.")
        return docs

    df = df.fillna("")

    for idx, row in df.iterrows():
        try:
            text = " | ".join(
                f"{str(col)}: {str(val)}"
                for col, val in row.items()
                if str(val).strip() != ""
            )

            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "row": idx,
                            "encoding": loaded_encoding
                        }
                    )
                )
        except Exception as e:
            print(f"⚠️ Error processing row {idx} in {file_path}: {e}")
            continue

    if not docs:
        print(f"⚠️ No valid documents extracted from {file_path}")
    
    return docs



def load_excel(file_path):
    docs = []
    xls = pd.ExcelFile(file_path)

    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df.fillna("")

        for idx, row in df.iterrows():
            text = " | ".join(
                f"{str(col)}: {str(val)}"
                for col, val in row.items()
                if str(val).strip() != ""
            )

            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "sheet": sheet,
                            "row": idx
                        }
                    )
                )
    return docs


def load_documents():
    documents = []

    for file in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file)

        if file.lower().endswith(".pdf"):
            documents.extend(load_pdf(file_path))

        elif file.lower().endswith(".csv"):
            documents.extend(load_csv(file_path))

        elif file.lower().endswith((".xls", ".xlsx")):
            documents.extend(load_excel(file_path))

    return documents


def main():
    print("📂 Loading documents...")
    documents = load_documents()

    if not documents:
        print("❌ No documents found")
        return

    print(f"✅ Loaded {len(documents)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTORSTORE_DIR)

    print("✅ Vector database created successfully")


if __name__ == "__main__":
    main()
