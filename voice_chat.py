from voice_input import listen_from_mic
from voice_output import speak
import chatbot  # your existing chatbot logic

def run_voice_chat():
    print("🎙️ Voice Chatbot Ready. Say 'exit' to stop.")

    while True:
        query = listen_from_mic()
        if not query or query.strip() == "":
            print("Didn't catch that. Please say again.")
            continue
        

        EXIT_KEYWORDS = ["exit", "quit", "stop", "goodbye", "bye"]

        if any(word in query.lower() for word in EXIT_KEYWORDS):
            speak("Hari Bol. Goodbye.")
            break

        

        
        
            
        

        answer = chatbot.process_query(query)  # you may need to expose this
        print(f"Bot: {answer}")
        speak(answer)

if __name__ == "__main__":
    run_voice_chat()
