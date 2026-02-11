import speech_recognition as sr
from voice_input import get_microphone

r = sr.Recognizer()

with get_microphone() as source:
    r.adjust_for_ambient_noise(source, duration=0.5)
    print("🎙️ Speak now...")
    audio = r.listen(source)

print("🗣️ You said:", r.recognize_google(audio))
