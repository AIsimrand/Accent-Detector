import streamlit as st
import os
from utils import download_video, extract_audio, classify_accent

st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🎙️ English Accent Classifier")

st.markdown("""
This tool accepts a **public video URL (MP4 format)**, extracts the audio, and attempts to classify the speaker's **English accent** using speaker embedding comparison with ECAPA-TDNN.

Please input a direct video link (e.g., from Loom, Dropbox, Google Drive with direct link format).
""")
video_url = st.text_input("Enter public video URL (direct MP4 link):")

if st.button("Analyze"):
    if not video_url:
        st.error("Please enter a video URL.")
    else:
        with st.spinner("🔄 Downloading and processing video..."):
            try:
                video_path = download_video(video_url, filename="temp_video.mp4")
                audio_path = extract_audio(video_path, output_audio="temp_audio.wav")
                accent, confidence = classify_accent(audio_path)

                st.success(f"✅ Accent Detected: **{accent}**")
                st.progress(int(confidence))
                st.write(f"**Confidence Score:** {confidence:.2f}%")

                if confidence < 50:
                    st.warning("⚠️ The confidence is low. Please ensure the audio is clear and try again.")

                # Cleanup temp files
                os.remove(video_path)
                os.remove(audio_path)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

