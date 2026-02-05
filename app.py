"""
Hugging Face Spaces App - Combines FastAPI (for API endpoint) and Gradio (for demo UI)
This runs on port 7860 by default on HF Spaces
"""
import gradio as gr
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import base64
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.api.schemas import DetectionRequest, DetectionResponse, ErrorResponse
from src.api.inference import predict_pipeline, load_resources
from src.api.lid import identify_language

# ===================== FASTAPI SETUP =====================
api_app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated voice samples in Tamil, English, Hindi, Malayalam, Telugu.",
    version="1.0.0"
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
print("Loading models...")
load_resources()
print("Models loaded!")

# API Key from environment or default
API_KEY = os.getenv("API_KEY", "voicedetect_2026_secure_key")

SUPPORTED_LANGUAGES = {
    'tamil': 'Tamil', 'english': 'English', 'hindi': 'Hindi',
    'malayalam': 'Malayalam', 'telugu': 'Telugu'
}

def fix_base64_padding(b64_string: str) -> str:
    b64_string = b64_string.strip()
    padding_needed = len(b64_string) % 4
    if padding_needed:
        b64_string += '=' * (4 - padding_needed)
    return b64_string

@api_app.get("/")
def health_check():
    return {"status": "online", "model_loaded": True}

@api_app.post("/api/voice-detection", responses={
    200: {"model": DetectionResponse},
    400: {"model": ErrorResponse},
    403: {"model": ErrorResponse}
})
async def detect_voice(request: DetectionRequest, x_api_key: str = Header(None)):
    try:
        if not x_api_key or x_api_key != API_KEY:
            return JSONResponse(status_code=403, content={"status": "error", "message": "Invalid API key"})
        
        lang_normalized = request.language.lower().strip()
        if lang_normalized not in SUPPORTED_LANGUAGES:
            return JSONResponse(status_code=400, content={
                "status": "error", 
                "message": f"Unsupported language. Supported: Tamil, English, Hindi, Malayalam, Telugu"
            })
        
        language_name = SUPPORTED_LANGUAGES[lang_normalized]
        
        supported_formats = ['mp3', 'wav', 'flac', 'ogg', 'm4a']
        if request.audio_format.lower() not in supported_formats:
            return JSONResponse(status_code=400, content={
                "status": "error", "message": f"Unsupported format. Supported: {', '.join(supported_formats)}"
            })
        
        try:
            b64_fixed = fix_base64_padding(request.audio_base64)
            audio_bytes = base64.b64decode(b64_fixed)
        except Exception as e:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid Base64: {e}"})
        
        if len(audio_bytes) > 10 * 1024 * 1024:
            return JSONResponse(status_code=400, content={"status": "error", "message": "File too large (max 10MB)"})
        
        result = predict_pipeline(audio_bytes)
        
        return JSONResponse(status_code=200, content={
            "status": "success",
            "language": language_name,
            "classification": result['result'],
            "confidenceScore": round(result['confidence'], 2),
            "explanation": result['explanation']
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ===================== GRADIO SETUP =====================
def file_to_bytes(file):
    with open(file, "rb") as f:
        return f.read()

def analyze_audio(audio_file):
    if audio_file is None:
        return None, "No file uploaded.", "Unknown"
    
    try:
        audio_bytes = file_to_bytes(audio_file)
        result = predict_pipeline(audio_bytes)
        
        if result['result'] == "AI_GENERATED":
            scores = {"AI_GENERATED": result['confidence'], "HUMAN": 1 - result['confidence']}
        else:
            scores = {"HUMAN": result['confidence'], "AI_GENERATED": 1 - result['confidence']}
        
        lang_id = identify_language(audio_file)
        return scores, result['explanation'], lang_id
        
    except Exception as e:
        return None, str(e), "Error"

with gr.Blocks(title="AI Voice Detector") as demo:
    gr.Markdown("# 🕵️ AI Voice Detection System")
    gr.Markdown("Upload an audio file to check if it's Human or AI-generated.")
    gr.Markdown("### 🔗 API Endpoint: `/api/voice-detection`")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio")
            submit_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            result_label = gr.Label(label="Prediction")
            lang_label = gr.Textbox(label="Detected Language")
            explanation_box = gr.Textbox(label="Explanation", lines=3)
    
    submit_btn.click(fn=analyze_audio, inputs=[audio_input], outputs=[result_label, explanation_box, lang_label])

# ===================== MOUNT TOGETHER =====================
# Mount FastAPI inside Gradio
app = gr.mount_gradio_app(api_app, demo, path="/demo")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
