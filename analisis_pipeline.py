import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path

# ─── PATH CONFIG ────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR    = os.path.join(BASE_DIR, "data", "corpus", "audio")
LOG_DIR      = os.path.join(BASE_DIR, "log")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "data", "corpus", "transcripts")

WHISPER_CLI  = os.path.join(BASE_DIR, "models", "whisper.cpp", "build", "bin", "Release", "whisper-cli.exe")
WHISPER_MODEL = os.path.join(BASE_DIR, "models", "whisper.cpp", "models", "ggml-large-v3-turbo.bin")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# ─── GEMINI CONFIG ───────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

SYSTEM_PROMPT = (
    "Kamu adalah asisten AI multibahasa yang memahami code-switching "
    "antara Bahasa Indonesia, Inggris, dan Arab. "
    "Jawab secara natural dan ringkas dalam Bahasa Indonesia."
)

# ─── STT (DENGAN NORMALISASI FFmpeg OTOMATIS) ────────────────────────────────
def transcribe(audio_path: str) -> dict:
    if not os.path.exists(WHISPER_CLI):
        return {"text": "", "success": False, "error": f"whisper-cli tidak ditemukan: {WHISPER_CLI}"}
    if not os.path.exists(WHISPER_MODEL):
        return {"text": "", "success": False, "error": f"Model tidak ditemukan: {WHISPER_MODEL}"}

    import tempfile
    temp_dir = tempfile.gettempdir()
    normalized_audio_path = os.path.join(temp_dir, f"normalized_{os.path.basename(audio_path)}")

    try:
        # LAKUKAN REMUX/RESAMPLING DENGAN FFmpeg
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", normalized_audio_path],
            capture_output=True, check=True
        )
    except Exception as e:
        # Fallback ke berkas asli jika gagal
        print(f"  [FFMPEG WARNING] Gagal melakukan remux, menggunakan file asli: {str(e)[:100]}")
        normalized_audio_path = audio_path

    try:
        result = subprocess.run(
            [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", normalized_audio_path, "-l", "id", "-nt"],
            capture_output=True, text=True, timeout=45  
        )

        # Hapus berkas sementara hasil normalisasi
        if normalized_audio_path != audio_path and os.path.exists(normalized_audio_path):
            try:
                os.remove(normalized_audio_path)
            except:
                pass

        raw = result.stdout
        clean = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]", "", raw)
        clean = re.sub(r"\[\d{2}:\d{2}:\d{2}\.\d{3}\]", "", clean)
        text = " ".join(clean.split()).strip()
        return {"text": text, "success": True, "error": None}
    except subprocess.TimeoutExpired:
        return {"text": "", "success": False, "error": "Timeout pemrosesan (File kemungkinan rusak/tidak standar)"}
    except Exception as e:
        return {"text": "", "success": False, "error": str(e)}


# ─── LLM (ROTASI GANDA KUNCI & MODEL OTOMATIS) ────────────────────────────────
def generate_response_safe(transcript: str, mode: str = "preserve") -> dict:
    if mode == "normalize":
        prompt = f"{SYSTEM_PROMPT}\n\nNormalisasi ke Bahasa Indonesia baku, lalu jawab: {transcript}"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nJawab dengan mempertahankan pola bahasa: {transcript}"

    # Mengumpulkan Kumpulan API Key Gemini dari .env
    gemini_keys = []
    if os.getenv("GEMINI_API_KEY"):
        gemini_keys.append(os.getenv("GEMINI_API_KEY"))
    for i in range(1, 11):
        k = os.getenv(f"GEMINI_API_KEY_{i}")
        if k:
            gemini_keys.append(k)

    if not gemini_keys:
        return {"response": "", "success": False, "error": "Tidak ada API Key Gemini yang dikonfigurasi di .env"}

    # Urutan prioritas model yang ingin dicoba sesuai opsi Anda
    MODELS_POOL = [
        "models/gemma-4-26b-a4b-it",
        "models/gemma-4-31b-it",
        "gemini-2.5-flash"
    ]

    # Rotasi otomatis kunci Gemini
    for idx, g_key in enumerate(gemini_keys, 1):
        try:
            g_client = genai.Client(api_key=g_key)
        except Exception as e:
            print(f"  [ROTASI] Gagal inisialisasi client untuk Key ke-{idx}: {str(e)[:100]}. Beralih ke Key berikutnya...")
            continue

        key_failed_completely = False

        # Mencoba model-model secara bertingkat dari yang paling diprioritaskan
        for model_name in MODELS_POOL:
            if key_failed_completely:
                break

            max_retries = 3
            base_delay  = 5.5  # Jeda aman RPM

            for attempt in range(max_retries):
                try:
                    time.sleep(base_delay)
                    response = g_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    # Berhasil! Langsung kembalikan respons jawaban
                    return {"response": response.text.strip(), "success": True, "error": None}

                except Exception as e:
                    err = str(e)
                    
                    # 1. Jika terkena rate limit (429), lakukan penundaan waktu sebelum mencoba kembali
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        wait = (2 ** attempt) * 10
                        print(f"  [GEMINI Key {idx} - {model_name}] Rate limit. Mencoba ulang dalam {wait}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(wait)
                        continue
                    
                    # 2. Jika model tidak ditemukan (404) atau tidak didukung, langsung beralih mencoba model berikutnya
                    if "404" in err or "not found" in err.lower() or "not supported" in err.lower() or "not_found" in err:
                        print(f"  [GEMINI Key {idx}] Model {model_name} tidak ditemukan/didukung. Beralih ke model berikutnya...")
                        break
                    
                    # 3. Jika API key tidak valid / salah, tandai kunci ini sebagai gagal total dan langsung ganti Key berikutnya
                    if "api key not valid" in err.lower() or "invalid_argument" in err.lower() or "400" in err:
                        print(f"  [GEMINI Key {idx}] Kunci tidak valid: {err[:120]}. Beralih ke KUNCI CADANGAN berikutnya...")
                        key_failed_completely = True
                        break
                    
                    # 4. Error tidak dikenal lainnya, coba lagi
                    print(f"  [GEMINI Key {idx} - {model_name}] Error: {err[:120]}. Mencoba ulang...")
                    time.sleep(3)
            else:
                # Dijalankan jika 3x retry untuk model ini habis dan masih gagal
                print(f"  [GEMINI Key {idx}] Model {model_name} gagal setelah semua retry. Mencoba model alternatif...")

        print(f"  [ROTASI] Kunci Gemini ke-{idx} telah dicoba untuk semua model. Beralih ke kunci selanjutnya...")

    return {"response": "", "success": False, "error": "Seluruh kombinasi Kunci Gemini dan Model di dalam pool telah habis atau gagal"}


# ─── MAIN PIPELINE (WITH AUTO-RESUME) ────────────────────────────────────────
def run_pipeline(mode: str = "preserve", limit: int = None):
    # Kumpulkan semua audio
    audio_files = sorted(Path(AUDIO_DIR).rglob("*.wav"))
    if limit:
        audio_files = audio_files[:limit]

    total = len(audio_files)
    log_path  = os.path.join(LOG_DIR, f"pipeline_{mode}_progress.json")
    
    results = []
    processed_filenames = set()

    # Muat progres lama jika file log sudah ada
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                old_results = json.load(f)
                # Hanya masukkan hasil yang sukses diproses (STT sukses) ke dalam riwayat resume
                for r in old_results:
                    # OPTIMASI: Jika transkrip STT kosong, jangan anggap sukses agar diproses ulang dengan FFmpeg
                    if r.get("stt_success") and r.get("stt_transcript", "").strip() != "" and (r.get("llm_success") or r.get("llm_error") is None):
                        results.append(r)
                        processed_filenames.add(r["filename"])
            print(f"[RESUME] Berhasil memuat progres sebelumnya. Melompati {len(processed_filenames)} data.")
        except Exception as e:
            print(f"[WARNING] Gagal memuat log progres lama, memulai ulang: {e}")
            results = []

    print(f"\n{'='*60}")
    print(f"Memulai analisis pipeline: {total} audio | mode: {mode}")
    print(f"Selesai diproses        : {len(processed_filenames)} audio")
    print(f"Sisa file               : {total - len(processed_filenames)} audio")
    print(f"{'='*60}\n")

    stt_ok    = sum(1 for r in results if r.get("stt_success"))
    llm_ok    = sum(1 for r in results if r.get("llm_success"))
    stt_fail  = 0
    llm_fail  = sum(1 for r in results if not r.get("llm_success") and r.get("stt_success"))
    latencies = [r["latency"] for r in results if "latency" in r]

    for idx, audio_path in enumerate(audio_files, 1):
        fname = audio_path.name
        
        # SKIP jika file audio ini sudah berhasil diproses di run sebelumnya
        if fname in processed_filenames:
            continue

        print(f"[{idx:3}/{total}] {fname}")

        t_start = time.time()
        entry = {
            "index":      idx,
            "filename":   fname,
            "audio_path": str(audio_path),
            "mode":       mode,
        }

        # ── STT ──
        stt = transcribe(str(audio_path))
        entry["stt_success"]    = stt["success"]
        entry["stt_transcript"] = stt["text"]
        entry["stt_error"]      = stt["error"]

        if stt["success"]:
            stt_ok += 1
            print(f"  STT: {stt['text'][:70]}")

            txt_path = os.path.join(TRANSCRIPT_DIR, fname.replace(".wav", ".txt"))
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(stt["text"])
        else:
            stt_fail += 1
            print(f"  STT GAGAL: {stt['error']}")
            entry["llm_success"] = False
            entry["llm_response"] = ""
            entry["latency"] = round(time.time() - t_start, 2)
            results.append(entry)
            
            # Simpan berkala saat gagal
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            continue

        # ── LLM (Dengan Rotasi Otomatis) ──
        llm = generate_response_safe(stt["text"], mode=mode)
        entry["llm_success"]  = llm["success"]
        entry["llm_response"] = llm["response"]
        entry["llm_error"]    = llm["error"]

        if llm["success"]:
            llm_ok += 1
            print(f"  LLM: {llm['response'][:70]}")
        else:
            llm_fail += 1
            print(f"  LLM GAGAL: {llm['error']}")

        # ── Latency ──
        latency = round(time.time() - t_start, 2)
        entry["latency"] = latency
        latencies.append(latency)

        results.append(entry)

        # Simpan log setiap data yang selesai diproses (auto-save bertahap)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Ringkasan Final ──
    print(f"\n{'='*60}")
    print(f"RINGKASAN PIPELINE")
    print(f"{'='*60}")
    print(f"Total audio       : {total}")
    print(f"STT berhasil      : {stt_ok} | gagal: {stt_fail}")
    print(f"LLM berhasil      : {llm_ok} | gagal: {llm_fail}")
    print(f"Avg latency       : {round(sum(latencies)/len(latencies),2) if latencies else 0}s")
    print(f"Log kemajuan      : {log_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  default="preserve", choices=["preserve","normalize","translate"])
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah audio (untuk testing)")
    args = parser.parse_args()

    run_pipeline(mode=args.mode, limit=args.limit)