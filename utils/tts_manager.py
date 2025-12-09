import pyttsx3
import threading


class TTSManager:
    """Manages text-to-speech operations."""

    def __init__(self, rate=150, volume=0.5):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.stop_flag = False
        self.tts_thread = None

    def speak_text(self, text):
        """Convert text to speech."""
        self.stop_flag = False  # Reset the flag

        def tts_thread():
            """Thread for TTS playback."""
            try:
                self.engine.say(text)
                self.engine.startLoop(False)
                while not self.stop_flag and self.engine.isBusy():
                    self.engine.iterate()
                self.engine.endLoop()
            except Exception as e:
                self.engine.stop()
                print(f"TTS Error: {e}")

        self.tts_thread = threading.Thread(target=tts_thread, daemon=True)
        self.tts_thread.start()

    def stop_tts(self):
        """Stop ongoing TTS playback."""
        self.stop_flag = True
        if self.tts_thread:
            self.tts_thread.join()

    def cleanup(self):
        """Cleanup the TTS engine."""
        self.stop_tts()
        self.engine.stop()
