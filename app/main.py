from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.controllers import router
import uvicorn
import os

# Initialize App
app = FastAPI(
    title="Motorcycle Noise Radar API",
    description="API for detecting license plates of loud motorcycles using ESP32 and YOLO/OCR.",
    version="1.0.0"
)

# Ensure static directory exists
os.makedirs("static/images", exist_ok=True)

# Mount Static Files (to serve images via URL)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register Routes/Controllers
app.include_router(router)

if __name__ == "__main__":
    # Host 0.0.0.0 allows access from external devices (ESP32) on the same network
    print("🚀 Starting Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)