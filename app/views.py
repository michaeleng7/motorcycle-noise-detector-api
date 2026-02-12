from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services import DetectorService
from app.models import InfracaoResponse

router = APIRouter()

# Instantiates the service (loads models into memory)
service = DetectorService()

# In-memory list to simulate database
simulated_database = []

@router.post("/register_infraction", response_model=InfracaoResponse)
async def register_infraction(
    file: UploadFile = File(...), 
    decibels: float = Form(...)
):
    """
    Endpoint receive image and noise level from ESP32.
    Receive image file and decibel level, process it, and return the result.
    """
    try:
        image_content = await file.read()
        
        # Call business logic to process the infraction
        analysis = service.process_infraction(image_content, decibels)
        
        record = {
            "id": str(len(simulated_database) + 1),
            "license_plate": analysis['placa'],
            "decibels": decibels,
            "image_path": analysis['arquivo'],
            "confidence": 0.0,  # Placeholder
            "status": analysis['status']
        }
        
        # Save to database
        simulated_database.append(record)
        
        return record
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard/data")
def get_dashboard_data():
    """Returns all infractions for the dashboard"""
    return simulated_database