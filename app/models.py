from pydantic import BaseModel
from typing import Optional

# Response Model (View representation in API context)
class InfractionResponse(BaseModel):
    id: str
    license_plate: str
    decibels: float
    image_url: str
    confidence: float
    status: str  # e.g., "Processed", "No Motorcycle Detected"
    timestamp: Optional[str] = None


# ========== NEW MODELS FOR GOOGLE DRIVE INTEGRATION ==========

class VideoOnDriveRequest(BaseModel):
    """Request model for receiving occurrence_id + drive_code from AI service"""
    occurrence_id: Optional[str] = None
    drive_file_id: str
    metadata: Optional[dict] = None  # Optional metadata from the AI service


class VideoOnDriveResponse(BaseModel):
    """Response model for video processing acknowledgment"""
    status: str  # "received", "processing", "completed"
    drive_file_id: str
    message: str
    timestamp: Optional[str] = None


class PlateCodeRequest(BaseModel):
    """Request model to receive plate code acknowledgment from AI service"""
    plate_code: str
    confidence: float
    occurrence_id: str
    additional_info: Optional[dict] = None


class PlateCodeResponse(BaseModel):
    """Response model for plate code acknowledgment from backend"""
    status: str  # "received", "acknowledged", "error"
    message: str
    plate_code: str
    timestamp: Optional[str] = None