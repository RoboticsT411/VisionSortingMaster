"""
OPTIONAL EXTRA - Voice commands (speech recognition, offline).

Try it on its own first:
    python voice_commands.py

Say one of the words in WORDS. Each word the program recognises is printed.

Setup (once):
  1. Install the packages into your project's virtual environment:
         pip install vosk sounddevice
  2. Download the small English model "vosk-model-small-en-us-0.15" (about 40 MB)
     from https://alphacephei.com/vosk/models and unzip it into this folder, so that
     the folder vosk-model-small-en-us-0.15 sits next to this file.
  3. Plug in a microphone (a USB headset works best in a noisy lab).

How it works: Vosk is a neural-network speech recogniser that runs on your PC,
with no internet. Giving it a short list of words (a "grammar") makes it much
more reliable: it only has to decide WHICH of these words you said.

In your sorting program:
    from voice_commands import VoiceListener
    voice = VoiceListener()          # starts listening in the background
    word = voice.get()               # the newest word said, or None (never waits)
    voice.stop()

Safety: a voice command is only read BETWEEN robot moves. It is not an
emergency stop: the emergency stop is the robot's power switch.
"""
import json
import queue
import threading
import time
from pathlib import Path

# The words the recogniser listens for. Add or remove words here.
WORDS = ["red", "yellow", "green", "all", "go", "skip", "stop"]
HERE = Path(__file__).resolve().parent
SMALL_MODEL = "vosk-model-small-en-us-0.15"        # recommended: 40 MB, loads in seconds


def find_model():
    """The small model if it is here, otherwise any other vosk-model-* folder next to this file."""
    if (HERE / SMALL_MODEL).exists():
        return HERE / SMALL_MODEL
    others = sorted(p for p in HERE.glob("vosk-model-*") if p.is_dir())
    return others[0] if others else HERE / SMALL_MODEL


MODEL_DIR = find_model()
SAMPLE_RATE = 16000


class VoiceListener:
    """Listens to the microphone in a background thread and keeps the recognised words."""

    def __init__(self, words=WORDS, model_dir=MODEL_DIR, device=None):
        try:
            import sounddevice
            import vosk
        except ImportError:
            raise SystemExit("Voice commands need two packages: pip install vosk sounddevice")
        if not Path(model_dir).exists():
            raise SystemExit(f"Speech model not found: {model_dir}\n"
                             f"Download vosk-model-small-en-us-0.15 from https://alphacephei.com/vosk/models "
                             f"and unzip it next to voice_commands.py")
        vosk.SetLogLevel(-1)
        self.words = list(words)
        name = Path(model_dir).name
        print(f"Loading the speech model {name}..." +
              ("" if "small" in name else " (a large model: this can take a minute or more)"))
        model = vosk.Model(str(model_dir))
        if "small" in name:
            # Small models accept a word list ("grammar"): they only decide WHICH of these words was said.
            grammar = json.dumps(self.words + ["[unk]"])      # [unk] = anything else
            self._recogniser = vosk.KaldiRecognizer(model, SAMPLE_RATE, grammar)
        else:
            # Large models cannot take a word list: they hear any English, and only our words are kept below.
            self._recogniser = vosk.KaldiRecognizer(model, SAMPLE_RATE)
        print("Speech model ready.")
        self._audio = queue.Queue()
        self._words = queue.Queue()                 # (word, time) for each command word
        self.last_heard = None                       # (text, time, is_command): for the display
        self._running = True
        self._stream = sounddevice.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000, dtype="int16",
                                                  channels=1, device=device,
                                                  callback=lambda data, *_: self._audio.put(bytes(data)))
        self._stream.start()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        while self._running:
            try:
                data = self._audio.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._recogniser.AcceptWaveform(data):
                text = json.loads(self._recogniser.Result()).get("text", "").strip()
                if not text:
                    continue
                now = time.time()
                commands = [w for w in text.split() if w in self.words]
                for word in commands:
                    self._words.put((word, now))
                self.last_heard = (commands[-1] if commands else text, now, bool(commands))

    def get(self, since=None):
        """The newest command word, or None. Never waits; older unread words are dropped.
        since = a time.time() value: words heard before it are ignored (e.g. said during a robot move)."""
        word = None
        while not self._words.empty():
            w, t = self._words.get_nowait()
            if since is None or t >= since:
                word = w
        return word

    def heard_text(self):
        """One line for the screen: the last thing heard and how long ago."""
        if self.last_heard is None:
            return "Heard: nothing yet"
        text, t, is_command = self.last_heard
        ago = f"{time.time() - t:.0f} s ago"
        if is_command:
            return f'Heard: "{text}" ({ago})'
        if text == "[unk]":
            return f"Heard: something that is not a command ({ago})"
        return f'Heard: "{text}" - not a command ({ago})'


    def stop(self):
        self._running = False
        self._stream.stop()
        self._stream.close()


def main():
    voice = VoiceListener()
    print("Listening for:", ", ".join(WORDS), "  (Ctrl+C to stop)")
    try:
        while True:
            word, _ = voice._words.get()                       # wait for the next word
            print("heard:", word)
    except KeyboardInterrupt:
        pass
    finally:
        voice.stop()


if __name__ == "__main__":
    main()
