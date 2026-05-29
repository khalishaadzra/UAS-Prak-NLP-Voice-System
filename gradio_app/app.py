import os
import tempfile
import requests
import gradio as gr
import scipy.io.wavfile

def voice_chat(audio):
    if audio is None:
        return None, "Tidak ada audio yang direkam."

    sr, audio_data = audio

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sr, audio_data)
        audio_path = tmpfile.name

    try:
        with open(audio_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            response = requests.post("http://localhost:8000/voice-chat", files=files, timeout=120)
    except requests.exceptions.ConnectionError:
        return None, "Tidak bisa terhubung ke server. Pastikan uvicorn sudah berjalan."
    except requests.exceptions.Timeout:
        return None, "Server timeout. Coba lagi."

    if response.status_code == 200:
        output_audio_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
        with open(output_audio_path, "wb") as f:
            f.write(response.content)
        return output_audio_path, "Pipeline selesai. Silakan dengar jawaban dari asisten."
    else:
        try:
            detail = response.json().get("detail", response.text)
        except:
            detail = response.text
        return None, f"Error {response.status_code}: {detail}"


css = """
footer { display: none !important; }

body {
    background-color: #f8f6ff !important;
}

.gradio-container {
    background-color: #f8f6ff !important;
    max-width: 860px !important;
    margin: 0 auto !important;
    font-family: 'Segoe UI', sans-serif;
}

.hero {
    background: #ffffff;
    border: 1px solid #e8e0f5;
    border-left: 4px solid #9b6dff;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(155, 109, 255, 0.06);
}

.hero h1 {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e1030;
    margin: 0 0 8px 0;
}

.hero p {
    color: #5a4f72;
    font-size: 0.9rem;
    margin: 0 0 16px 0;
    line-height: 1.65;
}

.badge-row {
    display: flex;
    gap: 7px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}

.badge {
    background: #f0ebff;
    border: 1px solid #d5c5ff;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.76rem;
    color: #6b3fcf;
    font-weight: 500;
}

.pipeline-row {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.8rem;
    color: #8a7aa0;
}

.pipeline-step {
    background: #f0ebff;
    border: 1px solid #cbb8ff;
    border-radius: 6px;
    padding: 2px 11px;
    font-weight: 600;
    color: #7c3aed;
    font-size: 0.78rem;
}

.arrow { color: #c4b0f0; }

.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #9b6dff;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Base(
    primary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#f8f6ff",
    body_background_fill_dark="#f8f6ff",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#ede8f8",
    block_border_color_dark="#ede8f8",
    block_label_text_color="#9b6dff",
    block_label_text_color_dark="#9b6dff",
    block_title_text_color="#6b3fcf",
    block_title_text_color_dark="#6b3fcf",
    input_background_fill="#fdfbff",
    input_background_fill_dark="#fdfbff",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_dark="#7c3aed",
    button_primary_background_fill_hover="#6d28d9",
    button_primary_background_fill_hover_dark="#6d28d9",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    border_color_primary="#e0d7f8",
    shadow_drop="0 2px 10px rgba(155,109,255,0.08)",
)) as demo:

    gr.HTML("""
    <div class="hero">
        <h1>Multilingual Voice Assistant</h1>
        <p>
            Sistem percakapan berbasis suara yang mendukung
            <strong style="color:#7c3aed">code-switching</strong>
            antara Bahasa Indonesia, Inggris, dan Arab.
            Silahkan bicara! Sistem akan memahami campuran bahasa
            dan merespons kembali dalam bentuk suara.
        </p>
        <div class="badge-row">
            <span class="badge">Bahasa Indonesia</span>
            <span class="badge">English</span>
            <span class="badge">العربية</span>
            <span class="badge">Code-Switching</span>
        </div>
        <div class="pipeline-row">
            Pipeline:&nbsp;
            <span class="pipeline-step">STT</span>
            <span class="arrow">→</span>
            <span class="pipeline-step">LLM</span>
            <span class="arrow">→</span>
            <span class="pipeline-step">TTS</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">Rekam Pertanyaan</div>')
            audio_input = gr.Audio(
                sources="microphone",
                type="numpy",
                show_label=False
            )
            submit_btn = gr.Button("Kirim", variant="primary")

        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">Jawaban Asisten</div>')
            audio_output = gr.Audio(
                type="filepath",
                show_label=False
            )
            status_box = gr.Textbox(
                show_label=False,
                lines=2,
                interactive=False,
                placeholder="Status akan muncul setelah audio dikirim."
            )

    submit_btn.click(
        fn=voice_chat,
        inputs=audio_input,
        outputs=[audio_output, status_box]
    )

demo.launch()