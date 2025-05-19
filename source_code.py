# Install dependencies
!pip install transformers datasets torchaudio librosa jiwer noisereduce evaluate kagglehub
!pip install fsspec==2025.3.2
# %%
import os
import torch
import librosa
import noisereduce as nr
import soundfile as sf
import tarfile
import numpy as np
from glob import glob
import kagglehub
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
from torch.utils.data import Dataset
from evaluate import load
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
# Download datasets
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)

speech_commands_url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
librispeech_url = "http://www.openslr.org/resources/12/test-clean.tar.gz"

def download_and_extract(url, download_path, extract_path):
    if not os.path.exists(extract_path):
        if not os.path.exists(download_path):
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, download_path)
            print("Download complete.")
        print(f"Extracting {download_path} ...")
        with tarfile.open(download_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print("Extraction complete.")
    else:
        print(f"{extract_path} already exists, skipping download.")

import urllib.request
download_and_extract(speech_commands_url, os.path.join(data_dir, "speech_commands_v0.02.tar.gz"), os.path.join(data_dir, "speech_commands"))
download_and_extract(librispeech_url, os.path.join(data_dir, "test-clean.tar.gz"), os.path.join(data_dir, "LibriSpeech_test-clean"))
# %%
def load_audio(file_path, target_sr=16000):
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio

def noise_reduction(audio):
    noise_sample = audio[0:int(0.5 * 16000)]
    reduced = nr.reduce_noise(y=audio, y_noise=noise_sample, sr=16000)
    return reduced
def load_google_speech_commands(data_dir, max_samples=1000):
    audio_paths, transcripts = [], []
    class_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d != '_background_noise_']
    for class_dir in class_dirs:
        wav_files = glob(os.path.join(class_dir, "*.wav"))
        for wav in wav_files[:max_samples]:
            audio_paths.append(wav)
            transcripts.append(os.path.basename(class_dir).lower())
        if len(audio_paths) >= max_samples:
            break
    return audio_paths[:max_samples], transcripts[:max_samples]

def load_librispeech(base_dir, max_samples=1000):
    audio_paths, transcripts = [], []
    for root, dirs, files in os.walk(base_dir):
        txt_files = [f for f in files if f.endswith(".txt")]
        for txt_file in txt_files:
            with open(os.path.join(root, txt_file), "r") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) < 2:
                        continue
                    utt_id, transcript = parts
                    audio_file = os.path.join(root, utt_id + ".flac")
                    if os.path.isfile(audio_file):
                        audio_paths.append(audio_file)
                        transcripts.append(transcript.lower())
                    if len(audio_paths) >= max_samples:
                        return audio_paths, transcripts
            return audio_paths, transcripts

def preprocess_dataset(audio_paths, transcripts):
    dataset = []
    for path, transcript in zip(audio_paths, transcripts):
        audio = load_audio(path)
        dataset.append({"audio": audio, "transcript": transcript})
    return dataset

# %%
speech_commands_extract = os.path.join(data_dir, "speech_commands")
librispeech_extract = os.path.join(data_dir, "LibriSpeech_test-clean")

gsc_audio_paths, gsc_transcripts = load_google_speech_commands(speech_commands_extract, max_samples=1200)
libri_audio_paths, libri_transcripts = load_librispeech(librispeech_extract, max_samples=800)

all_audio_paths = gsc_audio_paths + libri_audio_paths
all_transcripts = gsc_transcripts + libri_transcripts

dataset = preprocess_dataset(all_audio_paths, all_transcripts)
train_samples, val_samples = train_test_split(dataset, test_size=0.1, random_state=42)
# %%
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.to(device)
model.train()

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

def prepare_batch(sample):
    input_values = processor(sample["audio"], sampling_rate=16000, return_tensors="pt").input_values[0]
    labels = processor(text=sample["transcript"], return_tensors="pt").input_ids[0]
    return {"input_values": input_values, "labels": torch.tensor(labels)}
# %%
class ASRDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return prepare_batch(self.samples[idx])

def collate_fn(batch):
    batch = list(filter(None, batch))
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True, padding_value=0.0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_values": input_values, "labels": labels}

train_dataset = ASRDataset(train_samples)
val_dataset = ASRDataset(val_samples)
# %%
wer_metric = load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}
# %%
training_args = TrainingArguments(
    output_dir="./wav2vec2-captcha-model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_dir='./logs',
    report_to="none"
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)
# %%
trainer.train()
# %%
def transcribe_audio(path):
    model.eval()
    audio = load_audio(path)
    audio = noise_reduction(audio)
    input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription.lower()
captcha_audio_path = "/content/001ZIZ.wav"
print(transcribe_audio(captcha_audio_path))
# %%
from huggingface_hub import login
login("YOURTOKEN-XXXXXXXXXXXXXXXXXXX")

# %%
model_path = "./captcha-solver"

model.save_pretrained(model_path)
processor.save_pretrained(model_path)

# %%
from huggingface_hub import HfApi, HfFolder, Repository

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
model.push_to_hub("captcha-solver")
processor.push_to_hub("captcha-solver")

# %%
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Trainer, TrainingArguments
model = Wav2Vec2ForCTC.from_pretrained("partthdhiman/captcha-solver").to(device)
processor = Wav2Vec2Processor.from_pretrained("partthdhiman/captcha-solver")

# %%
def transcribe(path):
    audio, _ = librosa.load(path, sr=16000)
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000).input_values.to(device)

    with torch.no_grad():
        logits = model(inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# %%
