# Multilingual Code-Switching Speech-to-Speech Customer Service System

Proyek ini merupakan sistem asisten suara digital berbasis *web* yang dirancang khusus untuk menerima, memproses, dan merespons ujaran bahasa campuran (*code-switching*) antara Bahasa Indonesia, Inggris, dan Arab secara alami (*end-to-end*). 

Sistem ini mengintegrasikan teknologi Speech-to-Text (STT) lokal, kecerdasan Large Language Model (LLM) di awan dengan ketahanan tinggi (*failover system*), serta sintesis suara Text-to-Speech (TTS) lokal berbahasa Indonesia.

## 👨‍💻 Identitas Praktikan
*   **Nama:** Khalisha Adzraini Arif
*   **NPM:** 2308107010031
*   **Kelas:** Praktikum NLP A
*   **Tugas:** Ujian Akhir Semester (UAS) Pemrosesan Bahasa Alami 

---

## 📌 Fitur Utama &

1.  **🎙️ STT dengan whisper.cpp & Normalisasi FFmpeg Otomatis (*On-the-Fly*):**
    *   Menggunakan implementasi C++ berkinerja tinggi `whisper.cpp` (model `ggml-large-v3-turbo.bin`) dengan akselerasi instruksi **AVX2** pada CPU untuk transkripsi super cepat.
    *   Dilengkapi modul pra-pemrosesan **FFmpeg** otomatis. Setiap file audio dari berbagai perangkat perekam yang tidak standar akan otomatis dikonversi ke format wajib Whisper (**16000Hz, Mono, 16-bit PCM**) sebelum ditranskripsi, menyelesaikan masalah transkrip kosong (*silent error*).
2.  **🧠 LLM dengan Sistem Rotasi Kunci & Model (*Dual-Layer Failover*):**
    *   Mendukung model utama instruksi aslab: **`models/gemma-4-26b-a4b-it`** dan **`models/gemma-4-31b-it`**.
    *   **Rotasi Kunci API otomatis:** Jika satu kunci API Gemini terkena batas kuota harian (RPD), sistem akan otomatis mencabut kunci tersebut dan beralih menggunakan kunci cadangan berikutnya (`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, dst.) dari akun Google yang berbeda tanpa menghentikan antrean.
    *   **Rotasi Model otomatis:** Jika model Gemma utama tidak tersedia/mengalami error, sistem otomatis menurunkan prioritas secara bertingkat ke model gemma alternatif, dan terakhir ke `gemini-2.5-flash` sebagai jaring pengaman utama.
3.  **🔊 TTS dengan Coqui TTS & Konversi G2P Indonesia:**
    *   Menggunakan model lokal *Indonesian-TTS VITS* (Wikidepia) dengan karakter suara laki-laki `"wibowo"`.
    *   Dilengkapi pemrosesan fonetis kustom (`text_to_phoneme`) untuk mengubah ejaan teks menjadi lambat fonetik IPA Indonesia agar hasil pelafalan terdengar sangat natural.
4.  **🧪 Skrip Evaluasi Otomatis (`analisis_pipeline.py`):**
    *   Mendukung fitur **Auto-Resume** (melompati file yang sudah sukses diproses jika program terhenti ditengah jalan).
    *   **Auto-Retry untuk Berkas Kosong:** Mampu mendeteksi berkas hasil transkrip yang kosong pada pengujian sebelumnya untuk diproses ulang menggunakan normalisasi FFmpeg.
    *   **Timeout Pengaman:** Membatasi waktu tunggu pemrosesan maksimal 45 detik agar sistem tidak menggantung selamanya pada berkas audio yang rusak.

---

## 🗂️ Struktur Proyek

```text
UAS-PRAKTIKUM-PEMROSESAN-BAHASA-ALAMI/
├── app/
│   ├── main.py                 # Endpoint utama FastAPI
│   ├── llm.py                  # Integrasi Gemini/Gemma API & riwayat obrolan
│   ├── stt.py                  # Transkripsi suara (whisper.cpp)
│   ├── tts.py                  # TTS dengan Coqui + G2P fonetik
│   └── coqui_tts/              # Aset model lokal Coqui TTS Indonesia
│       ├── checkpoint_1260000-inference.pth
│       ├── config.json
│       └── speakers.pth
├── data/
│   ├── corpus/
│   │   ├── audio/              # Dataset 571 file audio .wav (.gitignore)
│   │   └── transcripts/        # Hasil transkripsi otomatis .txt
│   └── manifests/
├── gradio_app/
│   └── app.py                  # Antarmuka web pengguna interaktif (Gradio)
├── log/
│   └── pipeline_preserve_progress.json   # Log kemajuan analisis
├── models/
│   └── whisper.cpp/            # Hasil clone whisper.cpp (build via VS C++ Release)
│       ├── build/              # Folder build kompilasi (.gitignore)
│       └── models/             # File model ggml-large-v3-turbo.bin (.gitignore)
├── .env                        # Berkas kunci API Gemini rahasia (.gitignore)
├── .gitignore               
├── requirements.txt            # Dependensi Python yang disesuaikan untuk Windows + Python 3.11
└── analisis_pipeline.py        # Skrip otomatisasi pengujian 571 audio

---

### ⚙️ Panduan Setup dan Menjalankan Proyek

#### 1. Prasyarat Sistem
*   **Sistem Operasi:** Windows 10/11
*   **Python Version:** Python 3.11 (Sangat disarankan agar kompatibel dengan seluruh pustaka ML)
*   **Alat Tambahan:** **FFmpeg** harus terinstal di sistem dan terdaftar di dalam variabel lingkungan `PATH` Windows.

#### 2. Setup Virtual Environment
Aktifkan terminal PowerShell di root folder proyek, lalu jalankan perintah berikut:

```powershell
# Membuat Virtual Environment baru versi 3.11
py -3.11 -m venv env

# Mengaktifkan Virtual Environment
env\Scripts\activate

# Menginstal seluruh dependensi Windows
pip install -r requirements.txt
```

#### 3. Konfigurasi Kunci API (`.env`)
Buat berkas bernama **`.env`** di root folder proyek, lalu masukkan kunci API Gemini dari Google AI Studio:

```env
GEMINI_API_KEY=AIzaSyA_kunci_akun_google_1
GEMINI_API_KEY_1=AIzaSyB_kunci_akun_google_2
GEMINI_API_KEY_2=AIzaSyC_kunci_akun_google_3
```

#### 4. Menjalankan Server Backend dan UI Gradio

*   **Terminal 1 (Backend FastAPI):**
    ```powershell
    env\Scripts\activate
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

*   **Terminal 2 (Frontend UI Gradio):**
    ```powershell
    env\Scripts\activate
    python gradio_app/app.py
    ```
    *Buka url lokal `http://127.0.0.1:7860` pada browser untuk mencoba merekam suara.*

---

### 🧪 Cara Menjalankan Analisis 571 Audio

Untuk melakukan pemrosesan otomatisasi terhadap seluruh dataset audio, jalankan perintah berikut di terminal:

```powershell
env\Scripts\activate
python analisis_pipeline.py
```

*Skrip secara otomatis akan menyimpan berkas log kemajuan di dalam folder `log/` dan hasil transkrip teks di folder `transcripts/` secara bertahap.*