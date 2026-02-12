from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services import LicensePlateRecognitionService
from app.models import InfractionResponse
from datetime import datetime

router = APIRouter()

# Initialize Service (Singleton pattern approach for model loading)
recognition_service = LicensePlateRecognitionService()

# Mock Database (In-memory storage)
mock_database = []

@router.post("/report_infraction", response_model=InfractionResponse)
async def report_infraction(
    file: UploadFile = File(...), 
    decibels: float = Form(...)
):
    """
    Endpoint to receive data from ESP32.
    Payload: Image File + Decibel Level (float).
    """
    try:
        # Read file content
        image_content = await file.read()
        
        # Execute Business Logic
        analysis_result = recognition_service.process_infraction(image_content, decibels)
        
        # Create Record
        new_record = {
            "id": str(len(mock_database) + 1),
            "license_plate": analysis_result['license_plate'],
            "decibels": decibels,
            "image_url": analysis_result['file_path'],
            "confidence": 0.0, # Placeholder for OCR confidence score
            "status": analysis_result['status'],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to 'Database'
        mock_database.append(new_record)
        
        return new_record
    
    except Exception as e:
        # Return generic error 500
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/dashboard/data")
def get_dashboard_data():
    """
    Returns all recorded infractions for the frontend dashboard.
    """
    return mock_database