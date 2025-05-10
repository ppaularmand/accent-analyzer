# English Accent Analyzer

A tool that analyzes English accents from video URLs (YouTube, Loom, or direct video links).

## Features

- Analyzes accents from video or audio sources
- Detects 8 major English accent types: American, British, Australian, Indian, Canadian, Irish, Scottish, and South African
- Provides confidence scores and detailed explanations
- Simple command-line interface and web UI options

## Quick Setup (1-2 minutes)

### Prerequisites

- Python 3.8+ 
- FFmpeg (required for audio processing)

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/ppaularmand/accent-analyzer.git
   cd accent-analyzer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**

   - **Windows**:
     - Download from [ffmpeg.org](https://ffmpeg.org/download.html)
     - Add to your PATH environment variable

   - **macOS**:
     ```bash
     brew install ffmpeg
     ```
     
   - **Ubuntu/Debian**:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```

## Usage

### Command-line Interface

Analyze an accent from a video URL:

```bash
python accent_analyzer.py https://www.youtube.com/watch?v=example
```

With additional options:

```bash
python accent_analyzer.py https://www.youtube.com/watch?v=example --output results.json --keep-files
```

### Web Interface

Start the web interface:

```bash
streamlit run app.py
```

Then open your browser to the URL shown in the terminal (typically http://localhost:8501)

## Example Output

```json
{
  "accent": "British",
  "confidence": 78.45,
  "explanation": "Speaker exhibits non-rhotic pronunciation and typical British vowel qualities...",
  "all_scores": {
    "British": 78.45,
    "American": 10.24,
    "Irish": 5.67,
    "Australian": 3.21,
    "Scottish": 1.12,
    "Canadian": 0.78,
    "Indian": 0.32,
    "South African": 0.21
  },
  "key_features": ["t_glottalization", "open_vowels", "dental_fricatives"],
  "processing_time": 12.54
}
```

## Requirements.txt

Create a `requirements.txt` file with the following dependencies:

```
numpy
torch
torchaudio
librosa
pytube
pydub
requests
scikit-learn
transformers
streamlit
```

## Note on Accuracy

This tool provides an estimation of accents based on linguistic features. Accuracy may vary based on audio quality, speaker clarity, and accent strength. Results should be considered as indicative rather than definitive.

## License

[MIT License](LICENSE)
