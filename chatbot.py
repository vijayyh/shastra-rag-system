import os
import os.path
import logging  # Added for logging
from dotenv import load_dotenv
from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Logging Setup ---
# Added for logging
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# --- End Logging Setup ---

# Load environment variables
load_dotenv()
BOT_NAME = "ShastraBot"

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

VECTORSTORE_DIR = "vectorstore"

def is_mindmap_query(query: str) -> bool:
    keywords = ["mindmap", "mind map", "outline", "tree", "structure", "map"]
    q = query.lower()
    return any(k in q for k in keywords)

def is_teacher_query(query: str) -> bool:
    keywords = ["teach", "step by step", "start from basics", "explain in detail"]
    q = query.lower()
    return any(k in q for k in keywords)


def format_chat_history(chat_history: list) -> str:
    """Formats the chat history into a string for the LLM prompt."""
    if not chat_history:
        return ""
    
    formatted_history = "\n\nPrevious Conversation:\n"
    for user_query, bot_answer in chat_history:
        formatted_history += f"User: {user_query}\n"
        formatted_history += f"Bot: {bot_answer}\n"
    
    return formatted_history

#Normalization of words before query expansion if needed
import re

def normalize_query(query: str) -> str:
    """
    Normalizes user query to handle typos, variants, and transliterations.
    This function is intentionally lightweight and deterministic.
    """
    q = query.lower().strip()

    # Remove punctuation (keep words only)
    q = re.sub(r"[^\w\s]", "", q)

    # Common Sanskrit / domain-specific normalizations
    replacements = {
        "arjun": "arjuna",
        "bhraman": "brahman",
        "brahman": "brahman",
        "brahmana": "brahman",
        "geeta": "gita",
        "bhagvad": "bhagavad",
        "krishnaa": "krishna",
        "krsna": "krishna",
        "atma": "atman",
        "aatma": "atman",
        "dharmaha": "dharma"
    }

    words = q.split()
    normalized_words = [replacements.get(w, w) for w in words]

    return " ".join(normalized_words)

#function foe voice Speech -> text
def process_query(query: str) -> str:
    # move the logic that generates `answer`
    return answer





def main():
    # --- Chat History ---
    # Stores the last 5 conversation turns (user query + bot answer)
    chat_history = []
    # --- End Chat History ---

    # Load embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # --- Usage Hints ---
    # Added for usage hints
    print("RAG Chatbot Ready with source citation")

    print(f"\n🙏 Welcome! I am {BOT_NAME}.")
    print("I can help you understand concepts using the available data.")
    print("You can ask normal questions or request mind-map style explanations.")
    print("Example Prompts:")
    print('  - Normal: "What is Dharma?"')
    print('  - Mindmap: "Give me a mind map of the main concepts in the Bhagavad Gita"')
    print('  - Teacher: "Teach me about the concept of Atman, starting from the basics"\n')

    print("Type 'exit' to quit\n")
    # --- End Usage Hints ---

    while True:
        query = input("You: ")
        normalized_query = normalize_query(query)
        logging.info(f"Normalized query: {normalized_query}")
        if query.lower() == "exit":
            break

        # --- Logging User Query ---
        # Added for logging
        logging.info(f"User query: {query}")
        # --- End Logging ---

        # --- Query Expansion and Retrieval Logic ---
        # This block enhances retrieval by expanding short or ambiguous queries.
        # It is designed to be safe and controlled, falling back to simple retrieval if conditions are not met.
        try:
            # 1. Initial Retrieval
            # First, try a direct retrieval with the user's original query.
            initial_docs = retriever.invoke(normalized_query)
            
            # 2. Trigger Conditions for Query Expansion
            # Expansion is triggered if the query is short (<= 3 words) or initial retrieval is weak (< 3 docs).
            # This avoids wasting resources on already good queries.
            expand_query = len(normalized_query.split()) <= 3 and len(initial_docs) < 3
            
            all_docs = initial_docs
            
            # SAFETY: Expansion only runs if the 'expand_query' flag is True.
            # Otherwise, the logic proceeds with only the 'initial_docs'.
            if expand_query:
                logging.info(f"Query expansion triggered for query: '{query}'")

                # 3. Generate Expanded Queries with LLM
                # The LLM is prompted to generate alternative search queries, not to answer the question.
                # This keeps the expansion focused on retrieval.
                expansion_prompt = f"""
                Based on the user's query, generate 3 to 5 related, alternative search queries that could help find relevant information in a vector database. The queries should be short, concise, and diverse. Do not answer the question. Only provide a list of queries separated by newlines.

                User Query: "{normalized_query}"

                Alternative Queries:
                """
                
                try:
                    expansion_response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[{"role": "user", "content": expansion_prompt}],
                        temperature=0.4
                    )
                    expanded_queries_str = expansion_response.choices[0].message.content.strip()
                    
                    # Filter out empty lines from the response
                    for q in expanded_queries_str.split("\n"):
                        q = q.strip()
                        if not q:
                            continue
                         # remove bullets / numbering
                        q = q.lstrip("-•0123456789. ").strip()
                        if len(q.split()) <= 8:
                            expanded_queries.append(q)

                    expanded_queries = expanded_queries[:4]  # hard limit


                    # Log expanded queries for internal review. They are NOT shown to the user.
                    logging.info(f"Expanded queries: {expanded_queries}")

                    # 4. Retrieve Docs for Expanded Queries
                    # The original query is always included.
                    # This ensures the user's original intent is never lost.
                    queries_to_search = expanded_queries
                    for eq in queries_to_search:
                        # Adding exception handling for individual query retrieval
                        try:
                            all_docs.extend(retriever.invoke(eq))
                        except Exception as e:
                            logging.error(f"Retrieval error for expanded query '{eq}': {e}")
                            continue # Continue to the next query

                except Exception as e:
                    # If the expansion call fails, log the error and proceed with the initial documents.
                    logging.error(f"LLM error during query expansion for query '{query}': {e}")
                    # Fallback to initial_docs is implicit as all_docs is already populated

            # 5. Deduplicate Documents
            # Merges results from original and expanded queries, removing duplicates.
            # This creates a richer context without redundancy.
            unique_docs = {}
            for doc in all_docs:
                # Create a unique key based on source and page content hash
                doc_key = (doc.metadata.get("source"), hash(doc.page_content))
                if doc_key not in unique_docs:
                    unique_docs[doc_key] = doc
            
            docs = list(unique_docs.values())
            
            if not docs:
                print("\nBot: This question is not covered in the available data.\n")
                continue
                
        except Exception as e:
            print("\nShastraBot: Sorry, I encountered an error while retrieving information. Please try again.")
            logging.error(f"Retrieval error for query '{query}': {e}")
            continue
        # --- End Query Expansion and Retrieval Logic ---


        # Build context from documents and history
        # Added history_context for conversation memory
        history_context = format_chat_history(chat_history)
        # doc_context = "\n\n".join(doc.page_content for doc in docs)
        MAX_DOCS = 6
        doc_context = "\n\n".join(doc.page_content for doc in docs[:MAX_DOCS])
        context = f"{history_context}\n\n{doc_context}"

        # Collect and log sources
        sources = set()
        for doc in docs:
            src = doc.metadata.get("source", "Unknown source")
            sources.add(src)
        
        # --- Logging Retrieved Sources ---
        # Added for logging
        logging.info(f"Retrieved sources: {', '.join(map(os.path.basename, sources))}")
        # --- End Logging ---

        # Define prompt
        if is_mindmap_query(query):
            # Neutral tone for factual mind-map generation. Greetings should be omitted to maintain a clean structure.
            prompt = f"""
            You are an assistant that creates a clear, hierarchical mind-map in text form.
            Your purpose is to structure the information from the context below in a neutral, factual manner.

            Rules for Structuring:
            - Do not add any greeting or introductory text. The output must be only the mind-map itself.
            - Use a tree-like or bulleted structure.
            - Organize the information concisely and clearly.
            - Only include information present in the provided context.
            - Maintain a neutral, objective tone.

            Context:
            {context}

            Topic to map:
            {query}

            Mind-map:
            """
        elif is_teacher_query(query):
            # ISKCON-style teacher mode: Applies a humble, encouraging tone, and may start with a gentle greeting.
            # The prompt is enhanced to enforce a clear, step-by-step structure.
            prompt = f"""
            You are a patient and encouraging teacher, in the style of a gentle ISKCON guide. Your goal is to explain the topic step-by-step, using only the provided context. It is natural for you to start with a warm, devotional greeting like "Hari Bol 🙏".

            Guiding Principles:
            - You may begin with a single, kind greeting.
            - Your explanation MUST be structured in a clear, step-by-step format. Use a numbered list (e.g., 1., 2., 3.) or distinct paragraphs with clear headings for each step.
            - Start with the most foundational ideas, creating a strong base for understanding.
            - Gently build upon each concept, layer by layer, using phrases like "According to the teachings...".
            - Remain strictly within the provided context. Do not introduce outside ideology.
            - If the context is insufficient, kindly state so after your greeting.
            - Maintain a tone of humility, encouragement, and respect. Avoid absolute claims.

            Context from the teachings:
            {context}

            Topic to be taught with care:
            {query}

            A patient, step-by-step explanation:
            """
        else:
            # Medium emotional tone and language softening rules applied. A light, optional greeting is permitted for warmth.
            prompt = f"""
            You are a knowledgeable and respectful assistant. Your goal is to answer questions factually, based strictly on the provided context. For a touch of warmth, you might occasionally begin your response with a brief, gentle greeting like "Hare Krishna 🙏".

            Guiding Principles:
            - Start with a single, optional greeting only if it feels natural.
            - Answer only with the information given in the context.
            - Present facts in a calm, clear, and neutral tone.
            - Use gentle phrasing (e.g., "It can be understood as..."). Avoid absolute words.
            - If the answer is not in the context, state so clearly and gently after the optional greeting.

            Context:
            {context}

            Question:
            {query}

            Answer:
            """

        # --- Error Handling for LLM Call ---
        # Added for error handling
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print("\nShastraBot: Sorry, I couldn't generate a response. There might be an issue with the AI model.")
            # Added for logging
            logging.error(f"LLM error for query '{query}': {e}")
            continue
        # --- End Error Handling ---

        # Print answer
        print("\nShastraBot:", answer)

        # Print sources
        print("\nSources used:")
        for src in sources:
            filename = os.path.basename(src)
            _, file_extension = os.path.splitext(filename)
            file_type = file_extension.replace('.', '').upper()
            if not file_type:
                file_type = "Unknown"
            print(f"- [{file_type}] {filename}")

        print("\n" + "-" * 50 + "\n")

        # --- Update Chat History ---
        # Added to store the latest interaction
        chat_history.append((query, answer))
        # Keep the history to the last 5 turns
        if len(chat_history) > 5:
            chat_history.pop(0)
        # --- End Update Chat History ---



if __name__ == "__main__":
    main()
