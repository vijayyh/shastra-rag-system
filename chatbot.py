import os
import os.path
import logging
import re
import json
from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Constants and Configuration
BOT_NAME = "ShastraBot"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# Retrieval and Context Configuration
RETRIEVER_SEARCH_K = 8
CONTEXT_MAX_DOCS = 6
HISTORY_MAX_TURNS = 5

# Query Expansion Configuration
QUERY_EXPANSION_WORD_THRESHOLD = 3
QUERY_EXPANSION_DOC_THRESHOLD = 3
QUERY_EXPANSION_MAX_QUERIES = 4

# --- Logging Setup ---
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load environment variables
load_dotenv()

# Validate API Key on Startup
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "❌ GROQ_API_KEY not found in .env file. "
        "Please create a .env file with: GROQ_API_KEY=your_key_here"
    )

# Initialize Groq client
client = Groq(api_key=API_KEY)

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


def sanitize_query(query: str) -> str:
    """
    Sanitizes user input to prevent prompt injection attacks.
    Removes potentially malicious patterns.
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")
    
    query = query.strip()
    
    # Enforce length limit
    max_length = 5000
    if len(query) > max_length:
        query = query[:max_length]
        logging.warning(f"Query truncated from {len(query)} to {max_length} chars")
    
    # Block system prompt injection patterns
    blacklist = [
        "<system>", "</system>", "ignore instructions", "you are now",
        "forget your instructions", "system prompt", "jailbreak"
    ]
    
    query_lower = query.lower()
    for pattern in blacklist:
        if pattern in query_lower:
            raise ValueError(f"⚠️ Suspicious pattern detected: {pattern}")
    
    return query


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

class Chatbot:
    def __init__(self):
        """Initializes the Chatbot, loading models and vector store."""
        logging.info("Initializing Chatbot...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
            self.client = client
            self.chat_history = []
            logging.info("Chatbot initialized successfully.")
        except Exception as e:
            logging.error(f"Chatbot initialization failed: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm_with_retry(self, content: str, temperature: float = 0.2):
        """Calls LLM with automatic retry on failure."""
        return self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
            timeout=30
        )

    def _get_expanded_queries(self, normalized_query: str) -> list[str]:
        """Generates alternative search queries using an LLM."""
        expansion_prompt = f"""
        Based on the user's query, generate 3 to 5 related, alternative search queries that could help find relevant information in a vector database. The queries should be short, concise, and diverse. Do not answer the question. Only provide a list of queries separated by newlines.

        User Query: "{normalized_query}"

        Alternative Queries:
        """
        try:
            expansion_response = self._call_llm_with_retry(expansion_prompt, temperature=0.4)
            expanded_queries_str = expansion_response.choices[0].message.content.strip()
            
            expanded_queries = []  # FIX: Initialize list before use
            for q in expanded_queries_str.split("\n"):
                q = q.strip()
                if not q:
                    continue
                q = q.lstrip("-•0123456789. ").strip() # remove bullets / numbering
                if q:
                    expanded_queries.append(q)

            logging.info(f"Expanded queries: {expanded_queries}")
            return expanded_queries[:QUERY_EXPANSION_MAX_QUERIES]
        except Exception as e:
            logging.error(f"LLM error during query expansion: {e}")
            return []

    def _retrieve_docs(self, query: str):
        """
        Retrieves documents from the vector store, with optional query expansion.
        """
        normalized_query = normalize_query(query)
        logging.info(f"Normalized query: {normalized_query}")

        # 1. Initial Retrieval
        initial_docs = self.retriever.invoke(normalized_query)
        
        # 2. Trigger Conditions for Query Expansion
        expand_query = (len(normalized_query.split()) <= QUERY_EXPANSION_WORD_THRESHOLD and 
                        len(initial_docs) < QUERY_EXPANSION_DOC_THRESHOLD)
        
        all_docs = initial_docs
        
        if expand_query:
            logging.info(f"Query expansion triggered for query: '{query}'")
            expanded_queries = self._get_expanded_queries(normalized_query)
            for eq in expanded_queries:
                try:
                    all_docs.extend(self.retriever.invoke(eq))
                except Exception as e:
                    logging.error(f"Retrieval error for expanded query '{eq}': {e}")

        # 3. Deduplicate Documents
        unique_docs = {}
        for doc in all_docs:
            doc_key = (doc.metadata.get("source"), hash(doc.page_content))
            if doc_key not in unique_docs:
                unique_docs[doc_key] = doc
        
        return list(unique_docs.values())

    def _construct_prompt(self, query: str, context: str) -> str:
        """Constructs the appropriate prompt based on the query type."""
        if is_mindmap_query(query):
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

        return prompt

    def process_query(self, query: str) -> tuple[str, set[str]]:
        """
        Processes a user query and returns the answer and sources.
        This is the main method to interact with the chatbot.
        """
        answer = "Sorry, I couldn't generate an answer."
        sources = set()

        try:
            # 0. Sanitize input (SECURITY)
            query = sanitize_query(query)
            logging.info(f"User query: {query}")

            # 1. Retrieve documents
            docs = self._retrieve_docs(query)
            if not docs:
                answer = "This question is not covered in the available data."
                return answer, sources

            # 2. Build context and collect sources
            history_context = format_chat_history(self.chat_history)
            doc_context = "\n\n".join(doc.page_content for doc in docs[:CONTEXT_MAX_DOCS])
            context = f"{history_context}\n\n{doc_context}"

            for doc in docs:
                sources.add(doc.metadata.get("source", "Unknown source"))
            logging.info(f"Retrieved sources: {', '.join(map(os.path.basename, sources))}")

            # 3. Construct prompt and get LLM response
            prompt = self._construct_prompt(query, context)
            
            response = self._call_llm_with_retry(prompt, temperature=0.2)
            answer = response.choices[0].message.content.strip()

            # 4. Update chat history
            self.chat_history.append((query, answer))
            if len(self.chat_history) > HISTORY_MAX_TURNS:
                self.chat_history.pop(0)

        except ValueError as e:
            # Input validation error
            logging.warning(f"Input validation error: {e}")
            answer = f"⚠️ Invalid input: {str(e)}"
        except Exception as e:
            logging.error(f"Error processing query '{query}': {e}", exc_info=True)
            answer = "Sorry, I encountered an error. Please try again."

        return answer, sources


def main():
    """Runs the text-based chat interface."""
    try:
        chatbot = Chatbot()
    except Exception as e:
        print(f"Fatal Error: Could not start the chatbot. Please check logs. Error: {e}")
        return

    print("RAG Chatbot Ready with source citation")
    print(f"\n🙏 Welcome! I am {BOT_NAME}.")
    print("I can help you understand concepts using the available data.")
    print("You can ask normal questions or request mind-map style explanations.")
    print("Example Prompts:")
    print('  - Normal: "What is Dharma?"')
    print('  - Mindmap: "Give me a mind map of the main concepts in the Bhagavad Gita"')
    print('  - Teacher: "Teach me about the concept of Atman, starting from the basics"\n')
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        answer, sources = chatbot.process_query(query)

        print("\nShastraBot:", answer)
        if sources:
            print("\nSources used:")
            for src in sources:
                filename = os.path.basename(src)
                print(f"- {filename}")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
