
## 🗣️ English Accent Classifier

A lightweight web app built with **Streamlit** that classifies a speaker's **English accent** (British, American, or Australian) from a public video URL. It uses audio embedding comparison with the **ECAPA-TDNN** model from [SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

---

### 🚀 Features

* Accepts **direct MP4 video links**
* Extracts audio using **MoviePy**
* Detects accent using **ECAPA-TDNN speaker embeddings**
* Compares against reference accents: British, American, Australian
* Returns predicted accent with a **confidence score**

---

### 📦 Tech Stack

* [Streamlit](https://streamlit.io/) – for the frontend UI
* [MoviePy](https://zulko.github.io/moviepy/) – for audio extraction
* [SpeechBrain](https://speechbrain.readthedocs.io/) – for ECAPA-TDNN speaker recognition
* [Scipy](https://scipy.org/) – for cosine similarity
* [Torchaudio / SoundFile](https://pysoundfile.readthedocs.io/) – for audio handling

---

### 🔧 Setup Instructions

1. **Clone the repo**

```bash
git clone https://github.com/your-username/english-accent-classifier.git
cd english-accent-classifier
```

2. **Install dependencies**

Make sure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

3. **Add Reference Audio Files**

Place these 3 labeled audio samples in the root directory:

* `eng.wav` – British English speaker
* `usa.wav` – American English speaker
* `aus.wav` – Australian English speaker

> These are used as reference embeddings for comparison.

4. **Run the app locally**

```bash
streamlit run app.py
```

---

### 🌐 Streamlit Cloud Deployment

To deploy on [Streamlit Cloud](https://streamlit.io/cloud):

1. Make sure your repo has the following files:

   * `app.py`
   * `utils.py`
   * `requirements.txt`
   * `packages.txt` (with `ffmpeg`)

2. **`packages.txt`** (required for audio extraction via MoviePy):

```
ffmpeg
```

3. Add your 3 reference audio files (`eng.wav`, `usa.wav`, `aus.wav`) to the repository.

4. Deploy your GitHub repo on Streamlit Cloud.

---

### 📂 Project Structure

```
english-accent-classifier/
├── app.py                 # Streamlit frontend
├── utils.py               # Helper functions for download, audio, classification
├── eng.wav                # Reference accent - British
├── usa.wav                # Reference accent - American
├── aus.wav                # Reference accent - Australian
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies (ffmpeg)
```

---

### 📌 Limitations

* Works best with **clear single-speaker audio**
* Requires **public direct MP4 links** (e.g., Dropbox, Loom with raw link)
* Accent prediction is based on limited reference embeddings (3 samples)

---

### 📬 Future Enhancements

* Support more accent types (e.g., Indian, Irish, South African)
* Enable file upload instead of URL only
* Improve robustness to background noise and multi-speaker videos

