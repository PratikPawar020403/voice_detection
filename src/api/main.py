from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import base64
import os
import sys
from dotenv import load_dotenv

# Load env
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.api.schemas import DetectionRequest, DetectionResponse, ErrorResponse
from src.api.inference import predict_pipeline, load_resources

app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated voice samples in Tamil, English, Hindi, Malayalam, Telugu.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup
@app.on_event("startup")
async def startup_event():
    print("Initializing API...")
    load_resources()

# Auth - Load from environment
API_KEY = os.getenv("API_KEY", "12345")  # Default for local testing

# STRICT: Only accept full language names as per competition rules
# Short codes like 'en', 'ta', 'hi' are NOT allowed
SUPPORTED_LANGUAGES = {
    'tamil': 'Tamil',
    'english': 'English', 
    'hindi': 'Hindi',
    'malayalam': 'Malayalam',
    'telugu': 'Telugu'
}

def fix_base64_padding(b64_string: str) -> str:
    """
    Fix Base64 padding if missing.
    Some clients send Base64 without proper padding (== or =).
    """
    # Remove any whitespace
    b64_string = b64_string.strip()
    # Add padding if needed
    padding_needed = len(b64_string) % 4
    if padding_needed:
        b64_string += '=' * (4 - padding_needed)
    return b64_string

async def verify_api_key(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != API_KEY:
        return None  # Will be handled in endpoint
    return x_api_key

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": True}

@app.post("/api/voice-detection", responses={
    200: {"model": DetectionResponse},
    400: {"model": ErrorResponse},
    403: {"model": ErrorResponse}
})
async def detect_voice(request: DetectionRequest, x_api_key: str = Header(None)):
    """
    Detect whether a voice sample is AI-generated or Human.
    
    Required headers:
    - x-api-key: Your API key
    
    Supported languages: Tamil, English, Hindi, Malayalam, Telugu
    """
    try:
        # Validate API key first
        if not x_api_key or x_api_key != API_KEY:
            return JSONResponse(
                status_code=403,
                content={"status": "error", "message": "Invalid API key or malformed request"}
            )
        
        # STRICT language validation - only full names allowed
        lang_normalized = request.language.lower().strip()
        if lang_normalized not in SUPPORTED_LANGUAGES:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error", 
                    "message": f"Unsupported language: {request.language}. Supported languages are: Tamil, English, Hindi, Malayalam, Telugu (exact names only)"
                }
            )
        
        language_name = SUPPORTED_LANGUAGES[lang_normalized]
        
        # Validate audio format (only mp3 as per competition rules)
        if request.audio_format.lower() != 'mp3':
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Unsupported format: {request.audio_format}. Only mp3 is supported."
                }
            )
        
        # Decode base64 with padding fix
        try:
            # Fix padding if missing (some clients don't include proper padding)
            b64_fixed = fix_base64_padding(request.audio_base64)
            audio_bytes = base64.b64decode(b64_fixed)
        except Exception as decode_err:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Invalid Base64 string: {str(decode_err)}"}
            )
        
        # Validate audio size (max 10MB)
        if len(audio_bytes) > 10 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Audio file too large. Max 10MB."}
            )
            
        # Predict
        result = predict_pipeline(audio_bytes)
        
        # Build response matching competition specification exactly
        response = {
            "status": "success",
            "language": language_name,
            "classification": result['result'],  # AI_GENERATED or HUMAN
            "confidenceScore": round(result['confidence'], 2),
            "explanation": result['explanation']
        }
        
        return JSONResponse(status_code=200, content=response)
        
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
