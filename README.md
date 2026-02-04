# 🕵️ AI Voice Detection System

A production-ready API system to detect AI-generated voice samples vs Human speech, supporting **Tamil, English, Hindi, Malayalam, and Telugu** for the **AI-Generated Voice Detection** competition.


## 🎯 Project Overview

**Objective**: Build an API that accepts Base64-encoded MP3 audio and returns whether the voice is AI-generated or human, along with a confidence score and explanation.

**Supported Languages (Fixed)**:
- Tamil
- English
- Hindi
- Malayalam
- Telugu


## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python src/api/main.py
```
The API will be available at `http://localhost:8000`

### 3. (Optional) Start Gradio UI
```bash
python src/gradio_app.py
```
Interactive UI at `http://localhost:7861`


## 🔌 API Specification

### Endpoint
```
POST /api/voice-detection
```

### Headers
| Key | Value |
|-----|-------|
| `x-api-key` | Your API key |
| `Content-Type` | `application/json` |

### Request Body
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
}
```

### Request Fields
| Field | Description |
|-------|--------------|
| `language` | **Exact name required**: `Tamil`, `English`, `Hindi`, `Malayalam`, or `Telugu` |
| `audioFormat` | Always `mp3` |
| `audioBase64` | Base64-encoded MP3 audio |

### Success Response
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected"
}
```

### Error Response
```json
{
  "status": "error",
  "message": "Invalid API key or malformed request"
}
```

## 🔐 For Competition Endpoint Tester

### cURL Request Example
```bash
curl -X POST https://your-domain.com/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "language": "Tamil",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```


## 🛠 Deployment Options

### Option 1: ngrok (Quick Testing)
```bash
# Install ngrok
pip install pyngrok

# In terminal 1: Start API
python src/api/main.py

# In terminal 2: Create tunnel
ngrok http 8000
```
Use the `https://xxxx.ngrok.io` URL for the endpoint tester.

### Option 2: Render (Free Hosting)
1. Push code to GitHub
2. Connect to [render.com](https://render.com)
3. Create Web Service → Select repo
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`


## 📁 Project Structure

```
why/
├── data/
│   ├── raw/
│   │   ├── human/          # Human voice samples
│   │   └── ai/             # AI-generated samples
│   ├── processed/          # Preprocessed audio
│   └── features/           # Extracted features
├── models/                 # Trained model files
│   ├── dsp_model.pkl
│   └── dsp_cols.pkl
├── src/
│   ├── api/
│   │   ├── main.py         # FastAPI application
│   │   ├── inference.py    # Prediction pipeline
│   │   ├── schemas.py      # Pydantic models
│   │   └── lid.py          # Language identification
│   ├── features/
│   │   ├── extract_dsp.py  # DSP feature extraction
│   │   └── extract_embeddings.py
│   ├── config.py           # Configuration
│   ├── download_human_data.py
│   ├── generate_ai_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── gradio_app.py       # Gradio frontend
├── .env                    # Environment variables
├── requirements.txt        # Dependencies
├── EXPLANATION.md          # Detailed system explanation
└── README.md
```


## 📊 Model Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 95.7% |
| **Model Type** | Random Forest (Calibrated) |
| **Features** | 37 DSP features (MFCC, Spectral, Pitch) |
| **Full API Latency** | < 2 seconds |


## ⚙️ Configuration

### Environment Variables (.env)
```
API_KEY=your_secure_api_key
HF_TOKEN=your_huggingface_token
```

## ✅ Competition Compliance

| Requirement | Implementation |
|-------------|----------------|
| Accepts Base64 MP3 input | ✅ |
| Supports 5 languages | ✅ Tamil, English, Hindi, Malayalam, Telugu |
| Returns AI_GENERATED or HUMAN | ✅ |
| Returns confidenceScore (0-1) | ✅ |
| Returns explanation | ✅ |
| API Key authentication | ✅ x-api-key header |
| No hardcoded responses | ✅ Real ML model |
| No restricted external APIs | ✅ Only local model |


## 📜 License

This project is for competition purposes.
