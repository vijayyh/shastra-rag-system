from fastapi import FastAPI
from chatbot import Chatbot
from pydantic import BaseModel

app = FastAPI()
bot = Chatbot()

class Query(BaseModel): 
    message: str

@app.post("/chat")
def chat(query: Query):
    answer, sources = bot.process_query(query.message)
    return {
        "answer": answer,
        "sources": list(sources)
    }
