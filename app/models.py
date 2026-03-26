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
    """Request model for receiving a video file ID from backend colleague"""
    drive_file_id: str
    metadata: Optional[dict] = None  # Optional metadata from the backend


class VideoOnDriveResponse(BaseModel):
    """Response model for video processing acknowledgment"""
    status: str  # "received", "processing", "completed"
    drive_file_id: str
    message: str
    timestamp: Optional[str] = None


class PlateCodeRequest(BaseModel):
    """Request model to send detected plate code to backend colleague"""
    plate_code: str
    confidence: float
    video_id: str
    additional_info: Optional[dict] = None


class PlateCodeResponse(BaseModel):
    """Response model for plate code acknowledgment from backend"""
    status: str  # "received", "acknowledged", "error"
    message: str
    plate_code: str
    timestamp: Optional[str] = None