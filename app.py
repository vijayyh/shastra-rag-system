import gradio as gr
from chatbot import Chatbot

chatbot = Chatbot()

import os

def respond(message, history):
    answer, sources = chatbot.process_query(message)

    if sources:
        citation_text = "\n\n📚 Sources:\n"
        for i, src in enumerate(sources, 1):
            filename = os.path.basename(src)  # ← removes full path
            citation_text += f"{i}. {filename}\n"
        answer = answer + citation_text

    return answer


demo = gr.ChatInterface(respond)

if __name__ == "__main__":
    demo.launch()