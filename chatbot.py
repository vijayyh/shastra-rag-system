import os
import os.path
import logging
import re
import json
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Constants and Configuration
BOT_NAME = "ShastraBot"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
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
    CONFIDENCE_HIGH = 0.55
    CONFIDENCE_MEDIUM = 0.30

    DOMAIN_CONCEPT = """
    Hindu scriptures, Bhagavad Gita, Mahabharata, Ramayana,
    Vedas, Upanishads, dharma, karma, yoga, moksha,
    Krishna, Rama, Arjuna, spiritual philosophy
    """

    def __init__(self):
        """Initializes the Chatbot, loading models and vector store."""
        logging.info("Initializing Chatbot...")
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            vectorstore = FAISS.load_local(
                VECTORSTORE_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            self.DOMAIN_CONCEPT = """
            Hindu scriptures, Bhagavad Gita, Mahabharata, Ramayana,
            Vedas, Upanishads, dharma, karma, yoga, moksha,
            Krishna, Rama, Arjuna, spiritual philosophy
            """

            self.domain_vector = self.embeddings.embed_query(self.DOMAIN_CONCEPT)

            self.retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_SEARCH_K})
            # Build BM25 keyword index for hybrid retrieval
            self.docs = list(vectorstore.docstore._dict.values())
            self.tokenized_docs = [
            doc.page_content.lower().split() for doc in self.docs]
            self.bm25 = BM25Okapi(self.tokenized_docs)

            self.client = client
            self.chat_history = []

            self.last_entity = None
            self.domain_vector = self.embeddings.embed_query(self.DOMAIN_CONCEPT)

            
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

    def _keyword_search(self, query, top_k=5):
        """
        BM25 keyword search for hybrid retrieval.
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
        zip(self.docs, scores),
        key=lambda x: x[1],
        reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]


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
        
        # Hybrid retrieval: semantic + keyword
        keyword_docs = self._keyword_search(normalized_query)
        all_docs = initial_docs + keyword_docs

        
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
    

    def _is_context_relevant(self, query: str, docs) -> bool:
        """
        Checks whether retrieved docs are relevant enough to answer.
        Improved to support entity-based queries.
        """
        stopwords = {
            "who", "what", "is", "are", "the", "a", "an", "of",
            "in", "on", "at", "to", "for", "and", "why", "when"
        }
        query_words = [
            w for w in normalize_query(query).split()
            if w not in stopwords
        ]

        if not query_words:
            return False
        
        for doc in docs[:3]: # check top documents
            content = doc.page_content.lower()

            for word in query_words:
                if word in content:
                    return True  #strong entity match
            
        return False
    
    def _calculate_confidence(self, query: str, docs) -> float:
        """
        Calculates confidence score for retrieved context.

        Combines:
        - keyword/entity presence
        - frequency of query terms
        - document coverage
        """

        stopwords = {
            "who","what","is","are","the","a","an","of",
                "in","on","at","to","for","and","why","when"
        }

        query_terms = [
            w for w in normalize_query(query).split()
            if w not in stopwords
        ]

        if not query_terms or not docs:
            return 0.0

        term_hits = 0
        coverage_score = 0

        for doc in docs[:5]:
            text = doc.page_content.lower()

            for term in query_terms:
                if term in text:
                    term_hits += 1

        # how many documents contain query terms
        for term in query_terms:
            if any(term in d.page_content.lower() for d in docs[:5]):
                    coverage_score += 1

        term_density = term_hits / (len(query_terms) * min(len(docs), 5))
        coverage_ratio = coverage_score / len(query_terms)

        confidence = (term_density * 0.6) + (coverage_ratio * 0.4)

        return round(confidence, 3)

    def _is_answer_supported(self, answer: str, docs) -> bool:
        """
        Verifies whether the generated answer is supported
        by the retrieved context to prevent hallucinations.
        """

        if not answer or not docs:
            return True

        answer_words = set(answer.lower().split())

        ignore_words = {
            "the","is","are","was","were","and","of","to",
            "in","that","it","can","be","as","with","for",
            "from","this","these","those","have","has"
        }

        # extract meaningful terms
        key_terms = {
            w.strip(".,:;!?()[]")
            for w in answer_words
            if len(w) > 4 and w not in ignore_words
        }

        if not key_terms:
            return True

        context_text = " ".join(
            doc.page_content.lower() for doc in docs[:5]
        )

        matches = sum(1 for term in key_terms if term in context_text)

        support_ratio = matches / len(key_terms)

        return support_ratio >= 0.35


    
    DOMAIN_KEYWORDS = {
        "krishna","rama","ram","arjuna","sita","radha","hanuman",
        "shiva","vishnu","brahma","narayana","lakshmi","parvati",
        "gita","mahabharata","ramayana","vedas","upanishads",
        "dharma","karma","yoga","bhakti","atman","brahman",
        "moksha","samsara","avatar","kuru","pandava","kurukshetra"
    }

    def _is_domain_relevant(self, query: str, docs) -> bool:
        """
        Determines if query belongs to the scripture domain using
        semantic similarity instead of keyword lists.
        This scales to unlimited non-domain topics.
        """

        try:
            # embed user query
            query_vector = self.embeddings.embed_query(query)

            # cosine similarity (dot product since vectors are normalized)
            similarity = sum(q * d for q, d in zip(query_vector, self.domain_vector))

            # threshold determines domain relevance
            return similarity >= 0.38

        except Exception as e:
            logging.error(f"Domain detection error: {e}")
            return True   # fail-safe: allow instead of crash
        
    # SCRIPTURE FIDELITY GUARD -> layer 8
    DOCTRINAL_TERMS = {
        "dharma","karma","moksha","atman","brahman",
        "yoga","bhakti","jnana","samsara","guna",
        "satva","rajas","tamas","avatar","veda",
        "upanishad","sacred","cosmic","duty","righteousness",
        "liberation","self-realization","devotion"
    }

    def _check_doctrinal_fidelity(self, answer: str, docs) -> bool:
        """
        Ensures the response preserves doctrinal meaning
        and avoids oversimplified or distorted interpretations.
        """

        if not answer or not docs:
            return True

        answer_lower = answer.lower()

        # detect oversimplification phrases
        distortion_patterns = [
            "simply means",
            "just means",
            "only means",
            "basically means",
            "nothing but"
        ]

        if any(p in answer_lower for p in distortion_patterns):
            return False

        # ensure doctrinal terms present when relevant
        context_text = " ".join(
            doc.page_content.lower() for doc in docs[:5]
        )

        doctrinal_hits = sum(
            1 for term in self.DOCTRINAL_TERMS
            if term in answer_lower and term in context_text
        )

        # if doctrine discussed but no doctrinal language preserved
        if doctrinal_hits == 0 and any(term in context_text for term in self.DOCTRINAL_TERMS):
            return False

        return True
    

    def _is_ambiguous_followup(self, query: str) -> bool:
        """
        Detect ambiguous follow-up queries.
        Allow pronoun follow-ups if entity known.
        """
        words = set(query.lower().split())
        

        # pronouns that indicate dependency on previous context
        pronouns = {"it","this","that","they","them","he","she"}

        if words.intersection(pronouns):

            # if entity exists → NOT ambiguous
            if self.last_entity:
                return False

            return True
    

    def _extract_entity(self, query: str):
        """
        Extract key spiritual concept or figure from query.
        Prioritizes important domain terms.
        """

        domain_entities = {
        "dharma","karma","moksha","yoga","atman","brahman",
        "krishna","rama","arjuna","sita","hanuman",
        "vedas","gita","ramayana","mahabharata"
        }

        words = normalize_query(query).split()

        for w in words:
            if w in domain_entities:
                return w

        return None



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
            You are a knowledgeable and respectful assistant answering questions based strictly on the provided context.

            Response Guidelines:
            - Write in a natural, clear, and engaging way.
            - Avoid repetitive phrases or formulaic openings.
            - Present the information confidently but calmly.
            - If helpful, structure the answer with short paragraphs or bullet points.
            - Do not add information that is not present in the context.
            - If the answer is not in the context, say so clearly.

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

            # attach last entity to short follow-up questions
            if len(query.split()) <= 4 and self.last_entity:
                if not self._extract_entity(query):
                    query = query + " " + self.last_entity

            # Ambiguous Follow-up Detection:
            if self._is_ambiguous_followup(query):
                return(
                    "Your question seems to refer to something mentioned earlier. "
                    "Could you please clarify who or what you are referring to?",
                    set()                    
                )

            # 1. Retrieve documents
            docs = self._retrieve_docs(query)
            if not docs:
                answer = "This question is not covered in the available data."
                return answer, sources
            
            # --- Domain Boundary Guard ---
            if not self._is_domain_relevant(query, docs):
                logging.warning("Query outside domain.")
                return (
                    "This question appears to be outside the scope of the available spiritual texts.",
                    sources
                )
            
            # Update: Context Grounding Gaurd -->
            if not self._is_context_relevant(query,docs):
                logging.warning("Context not relevant enough. Refusing to answer.")
                return ( 
                    "I could not find a clear answer in the available texts. "
                    "Please try rephrasing or asking in more detail.",
                    sources
                )
            #confidence scoring
            confidence = self._calculate_confidence(query, docs)
            logging.info(f"Confidence score: {confidence}")

            #low confidence refusal
            if confidence < self.CONFIDENCE_MEDIUM:
                logging.warning("Low confidence answer refused.")
                return (
                    "I do not have enough reliable information in the available texts to answer this clearly.",
                    sources
                )

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

            # --- Output Validation Guard ---
            if not self._is_answer_supported(answer, docs):
                logging.warning("Answer contains unsupported content.")

                answer = (
                    "According to the available teachings:\n\n" + answer
                    )
                
            # --- Scripture Fidelity Guard ---
            if not self._check_doctrinal_fidelity(answer, docs):
                logging.warning("Doctrinal fidelity risk detected.")

                answer = (
                    "According to the teachings, the concept is more nuanced. "
                    "Based on the available texts:\n\n"
                    + answer
                )

            #adjust tone for moderate confidence
            if self.CONFIDENCE_MEDIUM <= confidence < self.CONFIDENCE_HIGH:
                answer = (
                    "Based on the available texts, it can be understood that:\n\n"
                    + answer
                )

            #Updated entity.
            entity = self._extract_entity(query)

            if entity:
                self.last_entity = entity

            # 4. Update chat history
            self.chat_history.append((query, answer))
            if len(self.chat_history) > HISTORY_MAX_TURNS:
                self.chat_history.pop(0)

            print("ENTITY:", self.last_entity)

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
