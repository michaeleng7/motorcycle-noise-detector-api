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