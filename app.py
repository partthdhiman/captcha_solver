import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
from pydub import AudioSegment
import io
import tempfile

# Load model and processor
@st.cache_resource
def load_model():
    model = Wav2Vec2ForCTC.from_pretrained("partthdhiman/captcha-solver")
    processor = Wav2Vec2Processor.from_pretrained("partthdhiman/captcha-solver")
    return model, processor

model, processor = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# App UI
st.title("🧠 Audio Captcha Solver")
st.write("Upload an MP3 or WAV file for transcription.")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.read()

        if uploaded_file.name.endswith(".mp3"):
            # Save the uploaded MP3 to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            # Convert MP3 to WAV using pydub
            audio = AudioSegment.from_file(tmp_path, format="mp3")
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav")
            wav_io.seek(0)
            waveform, sr = torchaudio.load(wav_io)

        else:  # WAV file
            waveform, sr = torchaudio.load(io.BytesIO(file_bytes))

        # Resample if not 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values.to(device)

        with st.spinner("Transcribing..."):
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])

        st.success("✅ Transcription complete:")
        st.write(f"**Text:** {transcription}")

    except Exception as e:
        st.error(f"🚫 Could not process audio file: {str(e)}")
