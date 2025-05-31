import requests
from moviepy.editor import VideoFileClip
import torchaudio
import torch
import os
from scipy.spatial.distance import cosine

# Load ECAPA-TDNN model
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Function to extract speaker embedding
# def get_embedding(file_path):
#     signal, fs = torchaudio.load(file_path)
#     with torch.no_grad():
#         embeddings = model.encode_batch(signal)
#     # Extract the first embedding and ensure it's 1-D
#     embedding = embeddings.squeeze().detach().cpu().numpy()
#     if embedding.ndim > 1:
#         embedding = embedding.mean(axis=0)
#     return embedding

import soundfile as sf
import torch

def get_embedding(file_path):
    signal, fs = sf.read(file_path)
    signal = torch.tensor(signal.T)  # transpose to [channels, time] if needed
    with torch.no_grad():
        embeddings = model.encode_batch(signal)
    return embeddings[0].squeeze()

# Step 1: Load reference embeddings
accent_refs = {
    "British": get_embedding("eng.wav"),
    "American": get_embedding("usa.wav"),
    "Australian": get_embedding("aus.wav"),

}


def download_video(url, filename="video.mp4"):
    """
    Downloads video from a public URL and saves it locally.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return filename
    else:
        raise Exception(f"Failed to download video. Status code: {response.status_code}")

def extract_audio(video_path, output_audio="audio.wav"):
    """
    Extracts audio from a given video file using moviepy.
    """
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio)
        return output_audio
    except Exception as e:
        raise Exception(f"Audio extraction failed: {str(e)}")

def classify_accent(audio_path):
    try:
        test_embedding = get_embedding(audio_path)
        similarities = {
            accent: 1 - cosine(test_embedding, ref_embedding)
            for accent, ref_embedding in accent_refs.items()
        }
        predicted_accent = max(similarities, key=similarities.get)
        confidence = similarities[predicted_accent] * 100
        return predicted_accent, confidence
    except Exception as e:
        raise Exception(f"Accent classification failed: {str(e)}")
