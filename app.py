import gradio as gr
from chatbot import Chatbot
import os

chatbot = Chatbot()

def respond(message, history):
    # Convert Gradio history to your chatbot's tuple format
    session_history = [
        (h[0], h[1])
        for h in history
        if h[0] and h[1]
    ]

    answer, sources = chatbot.process_query(message, session_history=session_history)

    if sources:
        citation_text = "\n\n📚 **Sources:**\n"
        for i, src in enumerate(sources, 1):
            filename = os.path.basename(src)
            citation_text += f"{i}. {filename}\n"
        answer = answer + citation_text

    return answer


demo = gr.ChatInterface(
    respond,
    title="📜 ShastraBot",
    description="Ask about Hindu scriptures — Gita, Vedas, Upanishads & more.",
    examples=[
        "What is the meaning of Dharma?",
        "Teach me about karma step by step",
        "Give me a mind map of the Bhagavad Gita",
        "Who is Krishna in the Mahabharata?",
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()