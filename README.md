# 🎙️ Persian Offline Transcription & Diarization

An end-to-end, privacy-focused tool for transcribing Persian (Farsi) audio and identifying different speakers (Diarization). This project leverages state-of-the-art AI models to provide high-accuracy text and speaker labeling completely offline.

---

## ✨ Key Features

* **🇮🇷 Persian Language Support:** Optimized for Farsi transcription using OpenAI's Whisper (via Faster-Whisper).
* **👥 Speaker Diarization:** Automatically separates and labels different voices (Speaker A, Speaker B, etc.).
* **🔒 Offline Processing:** No data leaves your machine. Perfect for sensitive or confidential recordings.
* **📂 Multiple Export Formats:** Download your results as **.TXT** transcripts or **.SRT** subtitle files.
* **📱 Modern UI:** Built with **Streamlit**, featuring a chat-like interface for reading conversations.
* **↔️ RTL Optimization:** Properly handles Right-to-Left (RTL) text display for Farsi.

---

## 🚀 Getting Started

### Prerequisites
* **Python:** 3.9 or higher.
* **FFmpeg:** Required for audio processing.
    * *Ubuntu:* `sudo apt install ffmpeg`
    * *Mac:* `brew install ffmpeg`
    * *Windows:* Download from [ffmpeg.org](https://ffmpeg.org/) and add to PATH.

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/persian-diarization.git](https://github.com/your-username/persian-diarization.git)
   cd persian-diarization

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   
3. **Model Setup (Two Options):**
   
   **Option A: Manual/Local (Offline)**
   
   **Place your pre-downloaded models in the ./models/ directory:**

   * Faster-Whisper model: ./models/whisper
   * SpeechBrain model: ./models/speechbrain 

   **Option B: Automatic Download (Hugging Face)**

    If you want the app to download the models automatically from the cloud, set your Hugging Face Token as an environment variable:
      ### Linux/Mac
           
           export HF_TOKEN="your_token_here"
      
      ### Windows (PowerShell)
           
          $env:HF_TOKEN="your_token_here"
     
### Running the App
    
    streamlit run app.py

## 🛠️ Tech Stack
 
  Component	        Technology
  Frontend	        Streamlit
  Transcription	    Faster-Whisper
  Diarization	      SpeechBrain
  Audio Engine	    PyDub & TorchAudio
  Clustering	      Scikit-Learn (K-Means)

## ⚡High-End Hardware Scaling

 If you have access to a powerful GPU (e.g., NVIDIA RTX 30/40 series), you can enhance this system by:

  1. Large-v3 Model: Upgrade to the large-v3 Whisper model for 15-20% better accuracy in complex Persian vocabulary.
  2. WhisperX Integration: Use forced alignment for word-level timestamps (perfect for high-end video subtitling).
  3. DeepFilterNet: Add a neural noise-reduction layer to clean noisy field recordings before transcription.
  4. LLM Post-Correction: Use a local LLM (like Llama-3-Persian) to fix punctuation and grammatical errors automatically.

## 📝 License

  This project is licensed under the MIT License.

## 🙏 Acknowledgments

  * OpenAI for the Whisper architecture.
  * SpeechBrain team for speaker recognition tools.
  * The Streamlit community for the amazing web framework.

  
