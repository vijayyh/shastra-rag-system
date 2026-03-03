import gradio as gr
from chatbot import Chatbot

chatbot = Chatbot()

def respond(message, history):
    answer, sources = chatbot.process_query(message)
    return answer

demo = gr.ChatInterface(respond)

if __name__ == "__main__":
    demo.launch()