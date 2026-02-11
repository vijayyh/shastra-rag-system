import speech_recognition as sr
import pyaudio
import logging

# Configure logging to append to the same file as the chatbot
logging.basicConfig(
    filename='chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_microphone():
    """Finds and returns the first available microphone, logging the process."""
    logging.info("Searching for an available microphone...")
    pa = pyaudio.PyAudio()
    mic_index = None
    mic_name = "Not Found"

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            mic_index = i
            mic_name = info.get('name', 'Unknown Mic')
            logging.info(f"Microphone found: '{mic_name}' (index: {i})")
            break

    pa.terminate()

    if mic_index is None:
        logging.error("No microphone could be found on the system.")
        raise RuntimeError("No microphone found")

    return sr.Microphone(device_index=mic_index)


def listen_from_mic():
    """Listens for audio from the mic, recognizes it, and logs outcomes."""
    recognizer = sr.Recognizer()

    # These 3 lines are CRITICAL
    recognizer.pause_threshold = 1.2      # seconds of silence = end of sentence
    recognizer.phrase_threshold = 0.3
    recognizer.non_speaking_duration = 0.8

    try:
        mic = get_microphone()
    except RuntimeError as e:
        print(f"❌ Microphone Error: {e}") # Keep for user feedback
        # The error is already logged in get_microphone(), no need to re-log.
        return None

    with mic as source:
        print("🎤 Listening... (speak your full question)")
        logging.info("Adjusting for ambient noise and starting to listen.")
        recognizer.adjust_for_ambient_noise(source, duration=0.8)

        try:
            audio = recognizer.listen(
                source,
                timeout=10,            # wait up to 10s to START speaking
                phrase_time_limit=None # No limit on how long the phrase can be
            )
        except sr.WaitTimeoutError:
            logging.info("No speech detected within the timeout period.")
            print("⏱️ No speech detected.") # Keep for user feedback
            return None

    try:
        logging.info("Sending audio to Google Speech Recognition API.")
        text = recognizer.recognize_google(audio)
        logging.info(f"Successfully recognized text: '{text}'")
        print(f"You (voice): {text}")
        return text.lower().strip()
    except sr.UnknownValueError:
        logging.warning("Google Speech Recognition could not understand the audio.")
        print("🤷 Could not understand audio.") # Keep for user feedback
        return None
    except sr.RequestError as e:
        logging.error(f"Could not request results from Google Speech Recognition service; {e}")
        print(f"❌ Speech service error: {e}") # Keep for user feedback
        return None