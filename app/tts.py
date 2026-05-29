import os
import re
import uuid
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COQUI_DIR = os.path.join(BASE_DIR, "coqui_utils")
if not os.path.exists(COQUI_DIR):
    COQUI_DIR = os.path.join(BASE_DIR, "coqui_tts")

COQUI_MODEL_PATH    = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")
COQUI_CONFIG_PATH   = os.path.join(COQUI_DIR, "config.json")
COQUI_SPEAKERS_PATH = os.path.join(COQUI_DIR, "speakers.pth")
COQUI_SPEAKER       = "wibowo"

_synthesizer = None


def text_to_phoneme(text: str) -> str:
    text = text.lower()
    
    # Digraf dulu — urutan penting
    text = text.replace("ng", "ŋ")
    text = text.replace("ny", "ɲ")
    text = text.replace("sy", "ʃ")
    text = text.replace("kh", "x")
    text = text.replace("gh", "ɡ")
    
    # Konsonan
    text = text.replace("c", "tʃ")
    text = text.replace("j", "dʒ")
    text = text.replace("q", "k")
    text = text.replace("v", "f")
    text = text.replace("z", "s")
    
    # g — ˈɡ di awal kata, ɡ di tengah
    text = re.sub(r'(?<![a-zəɔɛɪʊŋɲʃɡʒ])g', 'ˈɡ', text)
    text = text.replace("g", "ɡ")
    
    # y — jj di awal kata, j di tengah/akhir
    text = re.sub(r'(?<![a-zəɔɛɪʊŋɲʃɡʒ])y', 'jj', text)
    text = text.replace("y", "j")
    
    # Vokal
    text = text.replace("e", "ə")
    text = text.replace("o", "ɔ")
    
    # Bersihkan karakter non-vocab
    text = text.replace('\u201c', '').replace('\u201d', '').replace('\u2018', '').replace('\u2019', '')
    text = text.replace('-', ' ').replace('_', ' ')
    text = re.sub(r' +', ' ', text).strip()
    
    return text


def _get_synthesizer():
    global _synthesizer
    if _synthesizer is not None:
        return _synthesizer

    from TTS.utils.synthesizer import Synthesizer
    from TTS.tts.utils.speakers import SpeakerManager

    print(f"[TTS] Loading model dari: {COQUI_DIR}")

    synth = Synthesizer(
        tts_checkpoint=COQUI_MODEL_PATH,
        tts_config_path=COQUI_CONFIG_PATH,
        use_cuda=False,
    )

    if synth.tts_model.speaker_manager is None:
        print("[TTS] Injecting speaker_manager dari speakers.pth ...")
        sm = SpeakerManager()
        sm.load_ids_from_file(COQUI_SPEAKERS_PATH)
        synth.tts_model.speaker_manager = sm
        print(f"[TTS] Speaker tersedia: {list(sm.name_to_id.keys())[:5]}")

    print("[TTS] Model siap.")
    _synthesizer = synth
    return _synthesizer


def transcribe_text_to_speech(text: str) -> str:
    """
    Konversi teks ke audio menggunakan Coqui TTS + G2P Indonesia.
    Returns path ke file .wav, atau string '[ERROR] ...' jika gagal.
    """
    if not text or not text.strip():
        return "[ERROR] Teks kosong"

    for label, path in [
        ("model", COQUI_MODEL_PATH),
        ("config", COQUI_CONFIG_PATH),
        ("speakers", COQUI_SPEAKERS_PATH),
    ]:
        if not os.path.exists(path):
            return f"[ERROR] File {label} tidak ditemukan: {path}"

    try:
        import soundfile as sf

        synth = _get_synthesizer()

        # Konversi teks ke fonem dulu sebelum TTS
        phoneme_text = text_to_phoneme(text)
        print(f"[TTS] Original : {text[:80]}")
        print(f"[TTS] Phoneme  : {phoneme_text[:80]}")

        wav = synth.tts(text=phoneme_text, speaker_name=COQUI_SPEAKER)

        output_path = os.path.join(tempfile.gettempdir(), f"tts_{uuid.uuid4()}.wav")
        sf.write(output_path, wav, samplerate=22050)

        print(f"[TTS] Audio tersimpan: {output_path}")
        return output_path

    except Exception as e:
        print(f"[ERROR] TTS gagal: {e}")
        return f"[ERROR] {str(e)}"