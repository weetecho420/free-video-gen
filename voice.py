"""
Free text-to-speech in English and Tagalog/Filipino.

Main engine: edge-tts (Microsoft's free neural voices, no API key needed).
Backup engine: gTTS (Google Translate voice) if edge-tts is unavailable.
"""
import asyncio
import subprocess
import pathlib

# Friendly name -> (edge-tts voice id, gTTS language code)
VOICES = {
    "Tagalog – Blessica (female)": ("fil-PH-BlessicaNeural", "tl"),
    "Tagalog – Angelo (male)": ("fil-PH-AngeloNeural", "tl"),
    "English (Filipino accent) – Rosa (female)": ("en-PH-RosaNeural", "en"),
    "English (Filipino accent) – James (male)": ("en-PH-JamesNeural", "en"),
    "English (US) – Jenny (female)": ("en-US-JennyNeural", "en"),
    "English (US) – Guy (male)": ("en-US-GuyNeural", "en"),
    "English (US) – Aria (female)": ("en-US-AriaNeural", "en"),
    "English (US) – Andrew (male)": ("en-US-AndrewNeural", "en"),
}

DEFAULT_VOICE = "Tagalog – Blessica (female)"


def _edge_tts(text, voice_id, out_mp3, rate="+0%", pitch="+0Hz"):
    import edge_tts

    async def run():
        await edge_tts.Communicate(text, voice_id, rate=rate, pitch=pitch).save(out_mp3)

    try:
        asyncio.run(run())
    except RuntimeError:
        # Already inside a running event loop (Jupyter / Gradio)
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(run())


def _gtts(text, lang, out_mp3):
    from gtts import gTTS
    gTTS(text=text, lang=lang).save(out_mp3)


def audio_seconds(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(out) if out else 0.0


def make_voice(text, voice=DEFAULT_VOICE, out_wav="voice.wav", speed=0, pitch=0):
    """
    text  : the script the character says
    voice : one of VOICES keys
    speed : -50..+50 (percent slower/faster)
    pitch : -20..+20 (Hz lower/higher)
    Returns path to a 16 kHz mono WAV (the format the lip-sync model wants).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("The script is empty.")
    voice_id, gtts_lang = VOICES.get(voice, VOICES[DEFAULT_VOICE])
    mp3 = str(pathlib.Path(out_wav).with_suffix(".mp3"))

    try:
        _edge_tts(text, voice_id, mp3, rate=f"{int(speed):+d}%", pitch=f"{int(pitch):+d}Hz")
    except Exception as e:  # network hiccup etc.
        print(f"edge-tts failed ({e}); using gTTS backup voice instead.")
        _gtts(text, gtts_lang, mp3)

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3, "-ar", "16000", "-ac", "1", out_wav],
        check=True,
    )
    print(f"Voice saved: {out_wav} ({audio_seconds(out_wav):.1f}s)")
    return out_wav


if __name__ == "__main__":
    make_voice("Magandang araw! Salamat sa panonood.", out_wav="test_tl.wav")
    make_voice("Hello! Thanks for watching.", voice="English (US) – Jenny (female)", out_wav="test_en.wav")
