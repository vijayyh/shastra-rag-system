import gradio as gr
from chatbot import Chatbot

bot = Chatbot()

def chat_fn(user_input):
    if not user_input or not user_input.strip():
        return "Please enter a valid question."

    answer, sources = bot.process_query(user_input)

    if sources:
        src_text = "\n\nSources:\n" + "\n".join(
            f"- {s.split('/')[-1]}" for s in sources
        )
        answer += src_text

    return answer


with gr.Blocks(title="🕉️ ShastraBot") as demo:
    gr.Markdown(
        "## 🕉️ ShastraBot\n"
        "*AI Assistant for Vedic Scriptures*"
    )

    user_input = gr.Textbox(
        label="Your Question",
        placeholder="What are the duties of a human being according to the scriptures?",
        lines=2
    )

    output = gr.Textbox(
        label="ShastraBot Answer",
        lines=15
    )

    send_btn = gr.Button("Send")

    send_btn.click(
        chat_fn,
        inputs=user_input,
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
