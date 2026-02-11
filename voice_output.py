import pyttsx3
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_engine = None

def _init_engine():
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 170)
        _engine.setProperty("volume", 1.0)

def speak(text: str):
    if not text:
        return

    try:
        global _engine
        _init_engine()
        _engine.say(text)
        _engine.runAndWait()
        _engine.stop()  # 🔑 VERY IMPORTANT on Windows
    except Exception as e:
        logging.error(f"pyttsx3 error: {e}. Falling back to PowerShell.")
        # Reset the engine in case it's in a corrupted state
        _engine = None
        # Fallback: Windows PowerShell TTS
        ps_cmd = f'''
        Add-Type -AssemblyName System.Speech;
        $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
        $speak.Speak("{text}");
        '''
        try:
            subprocess.run(
                ["powershell", "-Command", ps_cmd],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as ps_e:
            logging.error(f"PowerShell TTS fallback failed: {ps_e}")
