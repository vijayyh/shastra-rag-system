"""
explorer.py — KnowledgeExplorer for ShastraBot
Each topic generates a scripture-grounded answer + 4 clickable follow-up
suggestions. Docs for suggestions are pre-fetched in background threads.
"""

import os
import json
import logging
import threading
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

LLM_MODEL = "llama-3.3-70b-versatile"


class KnowledgeExplorer:
    def __init__(self, chatbot_instance):
        """
        Args:
            chatbot_instance: existing Chatbot() — reuses _retrieve_docs()
                              and _call_llm_with_prompt().
        """
        self.chatbot = chatbot_instance
        self._prefetch_cache: dict = {}
        self._cache_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def explore(self, topic: str, path: list = None) -> tuple:
        """
        Args:
            topic : concept to explore, e.g. "Karma"
            path  : list of topics explored so far in this session

        Returns:
            (answer: str, suggestions: list[str])
        """
        path = path or []
        path_context = " → ".join(path) if path else ""

        # Use pre-fetched docs when available
        with self._cache_lock:
            docs = self._prefetch_cache.pop(topic, None)

        if docs is None:
            query = f"{topic} {path_context}".strip() if path_context else topic
            docs = self.chatbot._retrieve_docs(query)

        answer      = self._generate_answer(topic, docs, path_context)
        suggestions = self._generate_suggestions(topic, answer, path_context)

        # Pre-fetch docs for all 4 suggestions in the background
        full_path = " → ".join(path + [topic])
        self._prefetch_async(suggestions, full_path)

        return answer, suggestions

    # ── Answer generation ─────────────────────────────────────────────────────

    def _generate_answer(self, topic: str, docs: list, path_context: str) -> str:
        context_text = "\n\n".join(
            f"[Source {i+1}]:\n{d.page_content}"
            for i, d in enumerate(docs[:5])
        ) if docs else ""

        path_line = (
            f"\nThe student has been exploring: {path_context} → {topic}"
            if path_context else ""
        )

        prompt = f"""You are a wise Hindu scripture teacher.{path_line}

**Topic:** {topic}

**Scripture References:**
{context_text if context_text else "Draw from your knowledge of Hindu scriptures — Bhagavad Gita, Upanishads, Vedas, Mahabharata, and Vedic philosophy."}

**Instructions:**
- Explain '{topic}' in 3–5 focused paragraphs
- Ground the explanation in Hindu scriptures first; use broader knowledge only where context falls short
- Speak as a knowledgeable teacher — never say "according to the context"
- Never fabricate verse numbers or references unless certain
- End with a concise line formatted exactly as: **Essence:** <one-sentence distillation>

Begin your explanation of '{topic}':"""

        return self.chatbot._call_llm_with_prompt(prompt)

    # ── Suggestion generation ─────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _generate_suggestions(self, topic: str, answer: str, path_context: str) -> list:
        path_line = (
            f"Path so far: {path_context} → {topic}"
            if path_context else f"Starting topic: {topic}"
        )

        prompt = f"""A student just studied: **{topic}**
{path_line}

Generate exactly 4 natural follow-up questions they would want to explore next.

Rules:
- Each question must be SHORT (4–8 words maximum)
- Go deeper into '{topic}' or branch into key related subtopics
- Stay within Hindu scripture / Vedic philosophy
- Make each question meaningfully distinct
- Return ONLY a valid JSON array of exactly 4 strings, nothing else:
["q1", "q2", "q3", "q4"]"""

        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content.strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            if start >= 0 and end > start:
                suggestions = json.loads(raw[start:end])
                cleaned = [str(s).strip() for s in suggestions[:4]]
                while len(cleaned) < 4:
                    cleaned.append(f"More about {topic}")
                return cleaned
        except Exception as e:
            logging.warning(f"[Explorer] Suggestion generation failed: {e}")

        # Fallback
        return [
            f"Types of {topic}",
            f"{topic} in daily life",
            f"{topic} in Bhagavad Gita",
            f"How to practise {topic}",
        ]

    # ── Background pre-fetching ───────────────────────────────────────────────

    def _prefetch_async(self, suggestions: list, path_context: str) -> None:
        """Pre-fetch RAG docs for each suggestion so next click is instant."""

        def _fetch(suggestion: str) -> None:
            try:
                query = (
                    f"{suggestion} in context of {path_context}"
                    if path_context else suggestion
                )
                docs = self.chatbot._retrieve_docs(query)
                with self._cache_lock:
                    self._prefetch_cache[suggestion] = docs
                logging.info(f"[Explorer] Prefetched: '{suggestion}'")
            except Exception as e:
                logging.warning(f"[Explorer] Prefetch failed for '{suggestion}': {e}")

        for s in suggestions:
            threading.Thread(target=_fetch, args=(s,), daemon=True).start()