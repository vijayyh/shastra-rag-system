import os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

from groq import Groq

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Groq LLM Wrapper ----
class GroqLLM(LLM):
    @property
    def _llm_type(self):
        return "groq"

    def _call(self, prompt, stop=None):
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ CURRENTLY SUPPORTED
            messages=[
                {"role": "system", "content": "Answer strictly using the Bhagavad Gita context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=512
        )

        return response.choices[0].message.content


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = GroqLLM()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    print("📜 Bhagavad Gita RAG Chatbot Ready")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = qa_chain.invoke({"query": query})
        print("\nAnswer:", result["result"], "\n")


if __name__ == "__main__":
    main()
