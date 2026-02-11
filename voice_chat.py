from voice_input import listen_from_mic
from voice_output import speak
from chatbot import Chatbot
import logging

# Configure logging for voice chat
logging.basicConfig(
    filename='voice_chat.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_voice_chat():
    print("🎙️ Voice Chatbot Ready. Say 'exit' to stop.")

    # Initialize chatbot once
    try:
        bot = Chatbot()
        logging.info("Voice chatbot initialized successfully")
    except Exception as e:
        error_message = f"Fatal Error: Could not start the chatbot. Please check logs. Error: {e}"
        print(error_message)
        speak(error_message)
        logging.error(f"Chatbot initialization failed: {e}", exc_info=True)
        return

    EXIT_KEYWORDS = ["exit", "quit", "stop", "goodbye", "bye"]
    max_consecutive_errors = 3
    error_count = 0

    while True:
        max_mic_retries = 2
        query = None
        
        # Retry microphone input on transient errors
        for attempt in range(max_mic_retries):
            try:
                query = listen_from_mic()
                error_count = 0  # Reset error count on successful listen
                break
            except Exception as e:
                logging.error(f"Microphone error (attempt {attempt+1}/{max_mic_retries}): {e}")
                if attempt < max_mic_retries - 1:
                    speak("Having trouble hearing. Please repeat.")
                else:
                    speak("Microphone error. Please try again.")
        
        if not query:
            continue

        # Check for exit keywords
        if any(word in query for word in EXIT_KEYWORDS):
            print("👋 Exiting voice chatbot.")
            speak("Hari Bol. Goodbye.")
            logging.info("User initiated exit")
            break

        # Process query with error recovery
        try:
            answer, sources = bot.process_query(query)
            print(f"Bot: {answer}")
            
            # Try to speak the answer
            try:
                speak(answer)
            except Exception as e:
                logging.error(f"TTS error: {e}")
                print("(Audio output failed, but answer shown above)")
            
            error_count = 0  # Reset on successful processing
            logging.info(f"Query processed: {query[:50]}...")
            
        except Exception as e:
            error_count += 1
            logging.error(f"Query processing error ({error_count}/{max_consecutive_errors}): {e}", exc_info=True)
            print(f"❌ Error processing query: {e}")
            
            # Try to speak error message
            try:
                speak("Sorry, I encountered an error. Please try again.")
            except Exception as tts_e:
                logging.error(f"Failed to speak error message: {tts_e}")
            
            # Exit if too many consecutive errors
            if error_count >= max_consecutive_errors:
                print(f"❌ Exiting after {max_consecutive_errors} consecutive errors.")
                try:
                    speak("Too many errors. Goodbye.")
                except:
                    pass
                logging.warning(f"Exiting due to {max_consecutive_errors} consecutive errors")
                break

if __name__ == "__main__":
    run_voice_chat()
