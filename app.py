import gradio as gr
from chatbot import Chatbot

chatbot = Chatbot()

def respond(message, history):
    answer, sources = chatbot.process_query(message)

    if sources:
        citation_text = "\n\n📚 Sources:\n"
        for i, src in enumerate(sources, 1):
            citation_text += f"{i}. {src}\n"
        answer = answer + citation_text

    return answer

demo = gr.ChatInterface(respond)

if __name__ == "__main__":
    demo.launch()