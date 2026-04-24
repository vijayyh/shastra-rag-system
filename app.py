
import os
import gradio as gr
from chatbot import Chatbot
from explorer import KnowledgeExplorer



# ── Shared instances ──────────────────────────────────────────────────────────
chatbot = Chatbot()
explorer = KnowledgeExplorer(chatbot)


VEDIC_CSS = """
/* Force chatbot background — HF Spaces iframe safe */
.svelte-byatnx, [class*="svelte-"],
.wrap, .wrap > div, .wrap > div > div,
.chatbot .wrap, .chatbot .wrap *,
.message-wrap, .message-wrap * {
    background-color: #FDF3E3 !important;
    color: #2C1A0E !important;} 
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&display=swap');
.chatbot, [data-testid="chatbot"] {
    min-height: 500px !important;
}




@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmerShine {
    0%   { background-position: -300% center; }
    100% { background-position: 300% center;  }
}
@keyframes glowPulse {
    0%,100% { box-shadow: 0 2px 8px rgba(232,100,12,0.2); }
    50%      { box-shadow: 0 4px 20px rgba(232,100,12,0.55); }
}
@keyframes borderPulse {
    0%,100% { border-color: #D4A85A; }
    50%      { border-color: #E8640C; }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.93); }
    to   { opacity: 1; transform: scale(1);    }
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'EB Garamond', Georgia, serif !important;
    background-color: #FDF3E3 !important;
    color: #2C1A0E !important;
}

.tab-nav {
    background: #F5E6C8 !important;
    border-bottom: 2px solid #D4A85A !important;
    border-radius: 0 !important;
    padding: 6px 8px 0 !important;
    gap: 4px !important;
}
.tab-nav button {
    font-family: 'Cinzel', serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6B4423 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
    letter-spacing: 0.04em !important;
    transition: all 0.25s ease !important;
}
.tab-nav button.selected {
    background: linear-gradient(180deg, #E8640C 0%, #C9973A 100%) !important;
    color: #FFF !important;
    font-weight: 600 !important;
}
.tab-nav button:hover:not(.selected) {
    background: rgba(232,100,12,0.12) !important;
    color: #8B1A1A !important;
}


.message-wrap, .messages, .message-row,
[class*="message-wrap"], [class*="messages"],
[class*="message_wrap"] {
    background: transparent !important;
}

.message.bot, .message.bot *,
[data-testid="bot"], [data-testid="bot"] *,
.bot .message, .bot .message *,
[class*="bot"] .message, [class*="bot"] .message *,
.bubble-wrap.bot, .bubble-wrap.bot *,
div[data-role="bot"], div[data-role="bot"] * {
    background: #FBF0DC !important;
    color: #2C1A0E !important;
    border-color: #D4A85A !important;
}

.message.user,
[data-testid="user"],
.user .message,
[class*="user"] .message,
.bubble-wrap.user,
div[data-role="user"] {
    background: linear-gradient(135deg, #E8640C 0%, #C9973A 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
}
.message.user *, [data-testid="user"] *,
.user .message *, div[data-role="user"] * {
    color: #FFFFFF !important;
    background: transparent !important;
}

div[class^="svelte-"] > .message,
div[class*="svelte-"] > .message {
    color: #2C1A0E !important;
}

.message .prose, .message .markdown-body,
.message p, .message li, .message span,
.message h1, .message h2, .message h3,
.message strong, .message em, .message code {
    font-family: 'EB Garamond', serif !important;
    font-size: 17px !important;
    font-weight: 500 !important;
    line-height: 1.9 !important;
    color: inherit !important;
    background: transparent !important;
}
.message strong { font-weight: 600 !important; color: #8B1A1A !important; }
.message.user strong { color: #FFE8C8 !important; }
.message em { color: #9A6820 !important; font-style: italic !important; }
.message.user em { color: #FFE0B0 !important; }

.message-wrap > div, .message-row {
    animation: fadeSlideUp 0.45s ease both !important;
}





.message.bot, [data-testid="bot"], div[data-role="bot"] {
    border: 1px solid #D4A85A !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 16px !important;
    box-shadow: 0 2px 8px rgba(139,90,20,0.09) !important;
}

.message.user, [data-testid="user"], div[data-role="user"] {
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 16px !important;
}

.examples table td button,
.examples-row button,
.examples button,
.gr-samples-table button,
button[class*="example"],
[class*="examples"] button,
[class*="example-"] button {
    background: #FBF0DC !important;
    color: #5C1A00 !important;
    border: 1.5px solid #C9973A !important;
    border-radius: 8px !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.25s ease !important;
}
.examples table td button:hover,
.examples-row button:hover,
.examples button:hover,
[class*="examples"] button:hover {
    background: linear-gradient(135deg, #E8640C, #C9973A) !important;
    color: #FFFFFF !important;
    border-color: #E8640C !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(232,100,12,0.4) !important;
}
.examples .label-wrap span, .examples > .label span,
[class*="examples"] .label span {
    font-family: 'Cinzel', serif !important;
    font-size: 12px !important;
    color: #9A6820 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

button.primary, .primary,
button[variant="primary"], [variant="primary"] {
    font-family: 'Cinzel', serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #E8640C 0%, #C9973A 100%) !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(232,100,12,0.3) !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
button.primary::before, .primary::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(
        105deg,
        transparent 30%,
        rgba(255,255,255,0.45) 50%,
        transparent 70%
    ) !important;
    background-size: 300% 100% !important;
    background-position: -300% center !important;
    transition: background-position 0s !important;
}
button.primary:hover::before, .primary:hover::before {
    background-position: 300% center !important;
    transition: background-position 0.55s ease !important;
}
button.primary:hover, .primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(232,100,12,0.5) !important;
}
button.primary:active, .primary:active {
    transform: translateY(0) scale(0.97) !important;
    box-shadow: 0 2px 6px rgba(232,100,12,0.3) !important;
}

button.secondary, .secondary,
button[variant="secondary"], [variant="secondary"] {
    font-family: 'EB Garamond', serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #5C1A00 !important;
    background: #FBF0DC !important;
    border: 1.5px solid #C9973A !important;
    border-radius: 8px !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
button.secondary::before {
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(
        105deg,
        transparent 30%,
        rgba(201,151,58,0.35) 50%,
        transparent 70%
    ) !important;
    background-size: 300% 100% !important;
    background-position: -300% center !important;
    transition: background-position 0s !important;
}
button.secondary:hover::before {
    background-position: 300% center !important;
    transition: background-position 0.5s ease !important;
}
button.secondary:hover {
    transform: translateY(-2px) !important;
    border-color: #E8640C !important;
    color: #3D0D00 !important;
    box-shadow: 0 4px 14px rgba(201,151,58,0.4) !important;
}
button.secondary:active {
    transform: translateY(0) scale(0.97) !important;
}

button[aria-label*="Retry"], button[aria-label*="Undo"], button[aria-label*="Clear"],
.stop-btn {
    background: #F5E6C8 !important;
    color: #5C1A00 !important;
    border: 1px solid #C9973A !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    transition: all 0.22s ease !important;
}
button[aria-label*="Retry"]:hover, button[aria-label*="Undo"]:hover,
button[aria-label*="Clear"]:hover {
    background: #FBF0DC !important;
    border-color: #E8640C !important;
    transform: translateY(-1px) !important;
}

textarea, input[type="text"] {
    font-family: 'EB Garamond', serif !important;
    font-size: 17px !important;
    font-weight: 500 !important;
    color: #2C1A0E !important;
    background: #FDF8F0 !important;
    border: 1.5px solid #C9973A !important;
    border-radius: 8px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #E8640C !important;
    box-shadow: 0 0 0 3px rgba(232,100,12,0.15) !important;
    animation: borderPulse 2s ease infinite !important;
}
textarea::placeholder, input::placeholder {
    color: #B89860 !important;
    font-style: italic !important;
}

.prose, .markdown-body, .md, [class*="prose"] {
    font-family: 'EB Garamond', serif !important;
    font-size: 17px !important;
    font-weight: 500 !important;
    color: #2C1A0E !important;
    line-height: 1.9 !important;
}
.prose h1, .prose h2, .prose h3,
.markdown-body h1, .markdown-body h2, .markdown-body h3 {
    font-family: 'Cinzel', serif !important;
    color: #8B1A1A !important;
    border-bottom: 1px solid #D4A85A !important;
    padding-bottom: 4px !important;
    margin-top: 1.2em !important;
    font-weight: 600 !important;
}
.prose strong, .markdown-body strong {
    color: #8B1A1A !important;
    font-weight: 700 !important;
}
.prose em, .markdown-body em { color: #9A6820 !important; }
.prose li::marker, .markdown-body li::marker { color: #C9973A !important; }

#explorer-answer {
    background: #FBF0DC !important;
    border-left: 4px solid #E8640C !important;
    border-top: 1px solid #D4A85A !important;
    border-right: 1px solid #D4A85A !important;
    border-bottom: 1px solid #D4A85A !important;
    border-radius: 0 12px 12px 0 !important;
    padding: 16px 20px !important;
    animation: scaleIn 0.45s ease both !important;
    box-shadow: 0 3px 14px rgba(139,90,20,0.10) !important;
}
#explorer-answer *,
#explorer-answer .prose,
#explorer-answer .markdown-body,
#explorer-answer p,
#explorer-answer li,
#explorer-answer span,
#explorer-answer h1,
#explorer-answer h2,
#explorer-answer h3,
#explorer-answer strong,
#explorer-answer em {
    color: #2C1A0E !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 17px !important;
    font-weight: 500 !important;
    line-height: 1.95 !important;
    background: transparent !important;
}
#explorer-answer strong {
    color: #8B1A1A !important;
    font-weight: 700 !important;
}
#explorer-answer em { color: #9A6820 !important; }
#explorer-answer h1, #explorer-answer h2, #explorer-answer h3 {
    font-family: 'Cinzel', serif !important;
    color: #8B1A1A !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #D4A85A !important;
    padding-bottom: 4px !important;
    margin: 1em 0 0.4em !important;
}

#btn1, #btn2, #btn3, #btn4 {
    font-size: 16px !important;
    font-family: 'EB Garamond', serif !important;
    font-weight: 600 !important;
    color: #5C1A00 !important;
    background: #FBF0DC !important;
    border: 1.5px solid #C9973A !important;
    border-radius: 10px !important;
    padding: 13px 18px !important;
    line-height: 1.4 !important;
}
#btn1 { animation: fadeSlideUp 0.4s 0.05s ease both; }
#btn2 { animation: fadeSlideUp 0.4s 0.12s ease both; }
#btn3 { animation: fadeSlideUp 0.4s 0.19s ease both; }
#btn4 { animation: fadeSlideUp 0.4s 0.26s ease both; }

#breadcrumb-html, [id*="breadcrumb"] {
    animation: fadeSlideUp 0.35s ease both !important;
}

.block, .panel, .form {
    background: #FBF0DC !important;
    border: 1px solid #D4A85A !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(139,90,20,0.07) !important;
}

label span, .label-wrap span {
    font-family: 'Cinzel', serif !important;
    font-size: 12px !important;
    color: #6B4423 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #FDF3E3; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#E8640C, #C9973A);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: #CC4E00; }

footer { border-top: 1px solid #D4A85A !important; }
footer a, footer span {
    color: #9A6820 !important;
    font-family: 'EB Garamond', serif !important;
}
"""


# ── Chat handler ──────────────────────────────────────────────────────────────
def respond(message, history):
    session_history = [(h[0], h[1]) for h in history if h[0] and h[1]]
    answer, sources = chatbot.process_query(message, session_history=session_history)
    print("DEBUG ANSWER TYPE:", type(answer))
    print("DEBUG ANSWER:", answer)
    if sources:
        citation_text = "\n\n📚 **Sources:**\n"
        for i, src in enumerate(sources, 1):
            citation_text += f"{i}. {os.path.basename(src)}\n"
        answer = answer + citation_text
    return answer if isinstance(answer, str) else str(answer)


# ── Explorer handlers ─────────────────────────────────────────────────────────
def do_explore(topic, path, history):
    topic = topic.strip()
    if not topic:
        no_op = gr.update()
        return no_op, path, history, no_op, no_op, no_op, no_op, no_op, no_op

    answer, suggestions = explorer.explore(topic, path)
    # 🔥 HARD SAFETY
    if not isinstance(answer, str):
        answer = str(answer)
    if not isinstance(suggestions, list):
        suggestions = []
    suggestions = [str(s) for s in suggestions]
    new_path    = path + [topic]
    new_history = history + [{"topic": topic, "answer": answer}]
    suggestions = (suggestions + [""] * 4)[:4]

    return (
        gr.update(value=answer, visible=True),
        new_path,
        new_history,
        gr.update(value=_breadcrumb(new_path), visible=True),
        gr.update(value=suggestions[0], visible=bool(suggestions[0])),
        gr.update(value=suggestions[1], visible=bool(suggestions[1])),
        gr.update(value=suggestions[2], visible=bool(suggestions[2])),
        gr.update(value=suggestions[3], visible=bool(suggestions[3])),
        gr.update(value=topic),
    )

def on_suggestion(label, path, history):
    return do_explore(label, path, history)

def reset_explorer():
    off = gr.update(value="", visible=False)
    return off, [], [], off, off, off, off, off, gr.update(value="")

def _breadcrumb(path):
    if not path:
        return ""
    crumbs = (
        " &nbsp;<span style='color:#C9973A;font-size:18px;line-height:1'>›</span>&nbsp; "
        .join(
            f'<span style="font-family:Cinzel,serif;font-weight:600;color:#8B1A1A">{p}</span>'
            for p in path
        )
    )
    return (
        '<div style="'
        'padding:10px 16px;background:#F5E6C8;border-radius:8px;'
        'font-size:15px;font-family:EB Garamond,serif;color:#6B4423;'
        'border:1px solid #D4A85A;line-height:1.7;'
        'animation:fadeSlideUp 0.35s ease both'
        '">'
        '<span style="color:#C9973A;font-size:18px">🗺</span>'
        f'&nbsp;&nbsp;{crumbs}'
        '</div>'
    )


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    css=VEDIC_CSS,
    title="ShastraBot — Sacred Knowledge",
) as demo:

    gr.HTML("""
    <div style="
        text-align:center; padding:28px 20px 18px;
        border-bottom:2px solid transparent;
        border-image:linear-gradient(90deg,transparent,#C9973A,#E8640C,#C9973A,transparent) 1;
        background:#FBF0DC;
    ">
        <div style="font-size:36px;color:#C9973A;line-height:1.2;margin-bottom:8px;
                    animation:fadeIn 1s ease both">ॐ</div>
        <h1 style="
            font-family:'Cinzel',Georgia,serif; font-size:2.2em;
            font-weight:600; color:#8B1A1A; letter-spacing:0.08em;
            margin:0 0 8px; animation:fadeSlideUp 0.6s ease both;
        ">ShastraBot</h1>
        <p style="
            font-family:'EB Garamond',serif; color:#9A6820;
            font-size:16px; font-style:italic; margin:0; font-weight:500;
            animation:fadeSlideUp 0.7s 0.1s ease both;
        ">Wisdom from Hindu Scriptures · Bhagavad Gita · Vedas · Upanishads</p>
        <div style="
            margin-top:16px; height:1px;
            background:linear-gradient(90deg,transparent,#C9973A 30%,#E8640C 50%,#C9973A 70%,transparent);
        "></div>
    </div>
    """)

    with gr.Tabs():

        # ── Chat tab ──────────────────────────────────────────────────────────
        with gr.TabItem("💬 Chat"):
            gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=350), 
                examples=[
                    "What is the meaning of Dharma?",
                    "Teach me about karma step by step",
                    "Give me a mind map of the Bhagavad Gita",
                    "Who is Krishna in the Mahabharata?",
                ],
                title="",
                description=(
                    "Ask anything about Hindu scriptures — "
                    "Bhagavad Gita, Vedas, Upanishads, Mahabharata & Ramayana."
                ),
            )

        # ── Explorer tab ──────────────────────────────────────────────────────
        with gr.TabItem("🌳 Knowledge Explorer"):

            gr.HTML("""
            <div style="
                padding:16px 20px 12px; margin-bottom:14px;
                background:#FBF0DC; border-radius:10px;
                border:1px solid #D4A85A;
                box-shadow:0 2px 10px rgba(139,90,20,0.08);
                animation:fadeSlideUp 0.5s ease both;
            ">
                <h3 style="
                    font-family:'Cinzel',serif; color:#8B1A1A;
                    font-size:16px; font-weight:600; margin:0 0 8px;
                ">🌳 Knowledge Explorer</h3>
                <p style="
                    font-family:'EB Garamond',serif; color:#5C2E00;
                    font-size:16px; font-weight:500;
                    font-style:italic; margin:0; line-height:1.75;
                ">
                    Enter any concept from the scriptures to explore it deeply.
                    Follow the suggested paths to build your own tree of wisdom —
                    each step carries context forward.
                </p>
            </div>
            """)

            path_state    = gr.State([])
            history_state = gr.State([])

            with gr.Row():
                topic_input = gr.Textbox(
                    placeholder="e.g.  Karma · Prana · Maya · Dharma · Atman · Brahman …",
                    label="", scale=5, container=False, autofocus=True,
                )
                explore_btn = gr.Button("🔍 Explore", variant="primary", scale=1)
                reset_btn   = gr.Button("↺ Reset",   variant="secondary", scale=1)

            breadcrumb_display = gr.HTML("", visible=False, elem_id="breadcrumb-html")
            answer_display     = gr.Markdown("", visible=False, elem_id="explorer-answer")

            gr.HTML("""
            <div style="
                font-family:'Cinzel',serif; font-size:11px;
                letter-spacing:0.09em; color:#9A6820;
                margin:20px 0 10px; text-transform:uppercase;
            ">✦ &nbsp;Explore further</div>
            """)

            with gr.Row():
                btn1 = gr.Button("", visible=False, variant="secondary",
                                 size="lg", elem_id="btn1")
                btn2 = gr.Button("", visible=False, variant="secondary",
                                 size="lg", elem_id="btn2")
            with gr.Row():
                btn3 = gr.Button("", visible=False, variant="secondary",
                                 size="lg", elem_id="btn3")
                btn4 = gr.Button("", visible=False, variant="secondary",
                                 size="lg", elem_id="btn4")

            _outputs = [
                answer_display, path_state, history_state,
                breadcrumb_display, btn1, btn2, btn3, btn4, topic_input,
            ]

            explore_btn.click(do_explore,   [topic_input, path_state, history_state], _outputs)
            topic_input.submit(do_explore,  [topic_input, path_state, history_state], _outputs)
            reset_btn.click(reset_explorer, outputs=_outputs)

            for btn in [btn1, btn2, btn3, btn4]:
                btn.click(on_suggestion, [btn, path_state, history_state], _outputs)


# ── FIXED LAUNCH — works on HuggingFace Spaces ────────────────────────────────
import os

demo.queue().launch(share=True)