from pydantic import BaseModel, Field
from typing import Optional

class DetectionRequest(BaseModel):
    """
    Request body for voice detection API.
    Supports both full language names (Tamil, English, etc.) and codes (en, ta, etc.)
    """
    language: str = Field(..., description="Language: 'Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu' (or codes: 'en', 'ta', 'hi', 'ml', 'te')")
    audio_format: str = Field(..., alias="audioFormat", description="Audio format: always 'mp3'")
    audio_base64: str = Field(..., alias="audioBase64", description="Base64 encoded MP3 audio")
    
    class Config:
        populate_by_name = True  # Accept both field name and alias

class DetectionResponse(BaseModel):
    """
    Success response matching competition specification exactly.
    """
    status: str = Field(default="success", description="'success' or 'error'")
    language: str = Field(..., description="Language of the audio")
    classification: str = Field(..., description="'AI_GENERATED' or 'HUMAN'")
    confidenceScore: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    explanation: str = Field(..., description="Short reason for the decision")

class ErrorResponse(BaseModel):
    """
    Error response for invalid requests.
    """
    status: str = Field(default="error")
    message: str = Field(..., description="Error message")
