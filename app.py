import streamlit as st
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio
import io

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
st.title("🎧 Audio CAPTCHA Solver")
st.write("Upload an MP3 or WAV file for transcription.")

uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav"])

if uploaded_file is not None:
    try:
        # Load audio using torchaudio directly
        file_bytes = uploaded_file.read()
        audio_tensor, sample_rate = torchaudio.load(io.BytesIO(file_bytes))

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        # Prepare for model
        input_values = processor(audio_tensor.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

        with st.spinner("Transcribing..."):
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.decode(predicted_ids[0])

        st.success("✅ Transcription complete:")
        st.write(f"**Text:** {transcription}")

    except Exception as e:
        st.error(f"🚫 Could not process audio file: {str(e)}")
