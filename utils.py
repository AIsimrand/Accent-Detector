import requests
from moviepy.editor import VideoFileClip
from speechbrain.pretrained import SpeakerRecognition
import torchaudio
import os

import torchaudio
import torch
import os
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

# Load ECAPA-TDNN model
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Function to extract speaker embedding
def get_embedding(file_path):
    signal, fs = torchaudio.load(file_path)
    with torch.no_grad():
        embeddings = model.encode_batch(signal)
    # Extract the first embedding and ensure it's 1-D
    embedding = embeddings.squeeze().detach().cpu().numpy()
    if embedding.ndim > 1:
        embedding = embedding.mean(axis=0)
    return embedding

# Step 1: Load reference embeddings
# Example
accent_refs = {
    "British": get_embedding("eng.wav"),
    "American": get_embedding("usa.wav"),
    "Australian": get_embedding("aus.wav"),
    # Add more as needed
}


# Step 2: Compare unknown audio
# def detect_accent(unknown_audio_path):
#     test_embedding = get_embedding(unknown_audio_path)
#     similarities = {
#         accent: 1 - cosine(test_embedding, ref_embedding)
#         for accent, ref_embedding in accent_refs.items()
#     }
#     # Get the best match
#     predicted_accent = max(similarities, key=similarities.get)
#     confidence = similarities[predicted_accent]
#     return predicted_accent, confidence

# # Example usage
# predicted_accent, score = detect_accent("test_audio.wav")
# print(f"Predicted Accent: {predicted_accent} (Confidence: {score:.2f})")


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

# def classify_accent(audio_path):
#     """
#     Classifies the speaker's accent using a pre-trained language ID model.
#     Note: Replace with a more accurate accent-specific classifier for production use.
#     """
#     try:
#         classifier = SpeakerRecognition.from_hparams(
#             source="speechbrain/lang-id-commonlanguage_ecapa",
#             savedir="tmp/lang-id"
#         )
#         signal, fs = torchaudio.load(audio_path)
#         prediction = classifier.classify_file(audio_path)
#         label = prediction[0]  # Language label (e.g., English-US, English-UK)
#         score = float(prediction[1].item()) * 100  # Confidence score
#         return label, score
#     except Exception as e:
#         raise Exception(f"Accent classification failed: {str(e)}")

from scipy.spatial.distance import cosine

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
