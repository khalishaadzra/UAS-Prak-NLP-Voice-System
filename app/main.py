from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import tempfile
import sys

# Debug: Cetak sys.path untuk memeriksa jalur pencarian modul
print("sys.path:", sys.path)

try:
    from app.stt import transcribe_speech_to_text
    from app.llm import generate_response
    from app.tts import transcribe_text_to_speech
except ModuleNotFoundError as e:
    print(f"Error impor: {e}")
    raise

app = FastAPI()

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...)):
    """
    Endpoint untuk menerima file audio, memprosesnya, dan mengembalikan respons audio.
    Args:
        file: File audio yang diunggah dari frontend (format .wav)
    Returns:
        FileResponse: File audio hasil respons dari chatbot
    """
    # Validasi ekstensi file
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="File harus berformat .wav")

    # Baca konten file audio sebagai bytes
    audio_bytes = await file.read()
    print(f"Ukuran audio input: {len(audio_bytes)} bytes")
    if len(audio_bytes) < 100:  # Validasi ukuran minimal
        raise HTTPException(status_code=400, detail="File audio terlalu kecil atau kosong")

    # Langkah 1: Transkripsi audio ke teks menggunakan STT
    print(f"Langkah 1: Transkripsi audio ke teks menggunakan STT")
    transcribed_text = transcribe_speech_to_text(audio_bytes, file_ext=".wav")
    print(f"Transkripsi: {transcribed_text}")
    if transcribed_text.startswith("[ERROR]"):
        raise HTTPException(status_code=400, detail=transcribed_text)

    # Langkah 2: Kirim teks ke LLM untuk mendapatkan respons
    print(f"Langkah 2: Kirim teks ke LLM untuk mendapatkan respons")
    llm_response = generate_response(transcribed_text)
    print(f"Respons LLM: {llm_response}")
    with open("log/realtime_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\nSTT: {transcribed_text}\n")
        f.write(f"LLM: {llm_response}\n")
        f.write("-"*60 + "\n")
    if llm_response.startswith("[ERROR]"):
        raise HTTPException(status_code=400, detail=llm_response)

    # Langkah 3: Konversi respons teks ke audio menggunakan TTS
    print(f"Langkah 3: Konversi respons teks ke audio menggunakan TTS")
    audio_output_path = transcribe_text_to_speech(llm_response)
    print(f"Output TTS: {audio_output_path}")
    if audio_output_path.startswith("[ERROR]"):
        raise HTTPException(status_code=400, detail=audio_output_path)

    # Langkah 4: Validasi file audio
    print(f"Langkah 4: Validasi file audio")
    if not os.path.exists(audio_output_path) or os.path.getsize(audio_output_path) < 100:
        print(f"[ERROR] File audio tidak valid di {audio_output_path}")
        raise HTTPException(status_code=400, detail="File audio tidak valid atau kosong")

    print(f"File audio valid, mengembalikan: {audio_output_path}")
    return FileResponse(
        path=audio_output_path,
        filename="response.wav",
        media_type="audio/wav"
    )