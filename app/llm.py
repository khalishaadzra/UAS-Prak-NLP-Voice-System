import os
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"

# TODO: Ambil API key dari file .env
# Gunakan os.getenv("NAMA_ENV_VARIABLE") untuk mengambil API Key dari file .env.
# Pastikan di file .env terdapat baris: GEMINI_API_KEY=your_api_key
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# Prompt sistem yang digunakan untuk membimbing gaya respons LLM
system_instruction = """
Kamu adalah asisten AI multibahasa yang memahami code-switching antara Bahasa Indonesia, Inggris, dan Arab.
Jika input mengandung kata Arab, pertahankan beberapa kata Arab dalam responmu.
Jika input mengandung kata Inggris, pertahankan beberapa kata Inggris dalam responmu.
Jawab ringkas maksimal 2-3 kalimat.
Jangan mengulang pertanyaan pengguna.
Jika tidak tahu jawabannya, jujur saja katakan tidak tahu.
"""

# TODO: Inisialisasi klien Gemini dan konfigurasi prompt
# Gunakan genai.Client(api_key=...) untuk membuat klien.
# Gunakan types.GenerateContentConfig(system_instruction=...) untuk membuat konfigurasi awal.
# Jika ingin melihat contoh implementasi, baca dokumentasi resmi Gemini:
# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb
client = genai.Client(api_key=GOOGLE_API_KEY)
chat_config = types.GenerateContentConfig(system_instruction=system_instruction)
history_adapter = TypeAdapter(list[types.Content])

# Fungsi untuk menyimpan/memuat riwayat chat
def export_chat_history(chat) -> str:
    return history_adapter.dump_json(chat.get_history()).decode("utf-8")

def save_chat_history(chat):
    json_history = export_chat_history(chat)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(json_history)

def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return client.chats.create(model=MODEL, config=chat_config)
    
    if os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return client.chats.create(model=MODEL, config=chat_config)

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        json_str = f.read().strip()

    if not json_str:
        return client.chats.create(model=MODEL, config=chat_config)

    try:
        history = history_adapter.validate_json(json_str)
        return client.chats.create(model=MODEL, config=chat_config, history=history)
    except Exception as e:
        print(f"[ERROR] Gagal load history chat: {e}")
        return client.chats.create(model=MODEL, config=chat_config)

# Inisialisasi sesi chat saat aplikasi dimulai
chat = load_chat_history()

# Kirim prompt ke LLM dan kembalikan respons teks
def generate_response(prompt: str) -> str:
    try:
        response = chat.send_message(prompt)
        save_chat_history(chat)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"