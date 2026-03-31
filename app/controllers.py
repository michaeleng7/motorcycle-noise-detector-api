from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services import LicensePlateRecognitionService
from app.models import (
    InfractionResponse, 
    VideoOnDriveRequest, 
    VideoOnDriveResponse,
    PlateCodeRequest,
    PlateCodeResponse
)
from app.google_drive_service import GoogleDriveVideoService
from app.backend_integration_service import BackendIntegrationService
from datetime import datetime
import asyncio
import os

router = APIRouter()

# Initialize Service (Singleton pattern approach for model loading)
recognition_service = LicensePlateRecognitionService()

# Initialize Google Drive Service
try:
    google_drive_service = GoogleDriveVideoService("credentials.json")
except Exception as e:
    print(f"⚠️  Google Drive service initialization warning: {e}")
    google_drive_service = None

# Initialize Backend Integration Service for AI service
backend_integration_service = BackendIntegrationService(
    backend_url="http://0.tcp.sa.ngrok.io:12497"
)

# Mock Database (In-memory storage)
mock_database = []
video_processing_queue = {}  # Track videos being processed


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


# ========== NEW ENDPOINTS FOR GOOGLE DRIVE INTEGRATION ==========

@router.post("/video-on-drive", response_model=VideoOnDriveResponse)
async def video_on_drive(request: VideoOnDriveRequest):
    """
    ENDPOINT - Receive video file ID from backend  for processing.
    This endpoint is called by user's backend to notify that a new video is available on Google Drive for processing.
    
    Args:
        drive_file_id: The Google Drive file ID to download
        metadata: Optional metadata (e.g., timestamp, source info)
    
    Flow:
    1. Receive notification with Google Drive file ID
    2. Download video from Google Drive
    3. Extract frames at intervals
    4. Process each frame for license plate detection
    5. Send detected plate code back to backend  via POST /plate-code
    
    Response:
        - status: "received" when queued for processing
    """
    
    try:
        print(f"\n🎬 [VIDEO-ON-DRIVE] Received video notification from backend ")
        print(f"   Drive File ID: {request.drive_file_id}")
        
        # Check if Google Drive service is initialized
        if not google_drive_service or not google_drive_service.service:
            raise HTTPException(
                status_code=503,
                detail="Google Drive service not initialized. Check credentials.json setup."
            )
        
        # Store in processing queue
        video_processing_queue[request.drive_file_id] = {
            "status": "queued",
            "timestamp": datetime.now().isoformat(),
            "metadata": request.metadata
        }
        
        # Start processing asynchronously (background task)
        asyncio.create_task(
            process_video_from_drive(request.drive_file_id, request.metadata)
        )
        
        return VideoOnDriveResponse(
            status="received",
            drive_file_id=request.drive_file_id,
            message=f"Video queued for processing. File ID: {request.drive_file_id}",
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"❌ Error in /video-on-drive: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_video_from_drive(occurrence_id: str, drive_file_id: str, metadata: dict = None):
    """
    Background task to process video from Google Drive.
    
    Steps:
    1. Download video from Google Drive
    2. Extract frames
    3. Process frames with license plate recognition
    4. Send results back to backend 
    """
    
    try:
        print(f"\n⏳ [PROCESSING] Started processing video: {drive_file_id}")
        
        # Update status
        video_processing_queue[drive_file_id]["status"] = "downloading"
        
        # Step 1: Download video from Google Drive
        print(f"📥 Step 1: Downloading video from Google Drive...")
        video_bytes = google_drive_service.download_video_from_drive(drive_file_id)
        
        if not video_bytes:
            raise Exception("Failed to download video from Google Drive")
        
        # Step 2: Extract frames
        print(f"📹 Step 2: Extracting frames from video...")
        video_processing_queue[drive_file_id]["status"] = "extracting_frames"
        
        frames = google_drive_service.extract_frames_from_video(
            video_bytes, 
            frame_interval=30  # Extract every 30th frame
        )
        
        if not frames:
            raise Exception("No frames extracted from video")
        
        print(f"   📊 Total frames extracted: {len(frames)}")
        
        # Step 3: Process frames for license plate detection
        print(f"🔍 Step 3: Processing frames for license plate detection...")
        video_processing_queue[drive_file_id]["status"] = "analyzing"
        
        detected_plates = []
        
        for frame_array, frame_number in frames:
            try:
                # Convert frame to bytes for the recognition service
                _, frame_encoded = cv2.imencode('.jpg', frame_array)
                frame_bytes = frame_encoded.tobytes()
                
                # Process frame
                result = recognition_service.process_infraction(frame_bytes, decibels=0.0)
                
                if result['status'] == "Motorcycle detected and processed":
                    detected_plate = result.get('license_plate', 'UNKNOWN')
                    
                    if detected_plate != 'UNKNOWN':
                        detected_plates.append({
                            "plate": detected_plate,
                            "frame": frame_number,
                            "confidence": result.get('confidence_score', 0.0)
                        })
                        
                        print(f"   ✅ Frame {frame_number}: Plate detected - {detected_plate}")
            
            except Exception as frame_error:
                print(f"   ⚠️  Error processing frame {frame_number}: {frame_error}")
                continue
        
        # Step 4: Send best result back to backend 
        print(f"\n📤 Step 4: Sending results to backend ...")
        video_processing_queue[drive_file_id]["status"] = "sending_results"
        
        if detected_plates:
            # Sort by confidence and get the best one
            best_plate = max(detected_plates, key=lambda x: x['confidence'])
            
            print(f"   🏆 Best detected plate: {best_plate['plate']} (Confidence: {best_plate['confidence']:.2%})")
            
            # Send to backend 
            response = await backend_integration_service.send_plate_code_to_backend(
                plate_code=best_plate['plate'],
                confidence=best_plate['confidence'],
                video_id=drive_file_id,
                additional_info={
                    "frame_detected": best_plate['frame'],
                    "total_frames_analyzed": len(frames),
                    "all_detected_plates": detected_plates
                }
            )
            
            video_processing_queue[drive_file_id]["status"] = "completed"
            video_processing_queue[drive_file_id]["result"] = response
            
            print(f"✅ [COMPLETED] Video processing finished successfully")
            print(f"   Result: {response}")
        
        else:
            print(f"⚠️  No license plates were detected in the video")
            
            response = await backend_integration_service.send_plate_code_to_backend(
                plate_code="NOT_DETECTED",
                confidence=0.0,
                occurrence_id=occurrence_id,
                additional_info={
                    "frames_analyzed": len(frames),
                    "note": "No motorcycle or license plate detected in video"
                }
            )
            
            video_processing_queue[drive_file_id]["status"] = "completed_no_detection"
            video_processing_queue[drive_file_id]["result"] = response
    
    except Exception as e:
        print(f"❌ Error processing video {drive_file_id}: {str(e)}")
        video_processing_queue[drive_file_id]["status"] = "error"
        video_processing_queue[drive_file_id]["error"] = str(e)


@router.post("/plate-code", response_model=PlateCodeResponse)
async def plate_code_endpoint(request: PlateCodeRequest):
    """
    ENDPOINT - Receive acknowledgment that plate code was delivered to backend.
    
    This endpoint is called by user's backend  to acknowledge that the plate code sent from this service was received and processed by their backend.
    
    Args:
        plate_code: The license plate code
        confidence: Confidence score
        video_id: Google Drive file ID
        additional_info: Optional metadata
    
    Response:
        Acknowledgment that the endpoint was reached
    """
    
    try:
        print(f"\n✅ [PLATE-CODE ACK] Backend  acknowledged plate code: {request.plate_code}")
        
        # Log the acknowledgment
        ack_record = {
            "plate_code": request.plate_code,
            "video_id": request.video_id,
            "confidence": request.confidence,
            "timestamp": datetime.now().isoformat(),
            "additional_info": request.additional_info
        }
        
        # You could store this in a database or log file
        print(f"   📝 Recorded: {ack_record}")
        
        return PlateCodeResponse(
            status="received",
            message="Plate code acknowledgment received successfully",
            plate_code=request.plate_code,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        print(f"❌ Error in /plate-code endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video-processing-status/{drive_file_id}")
def get_video_processing_status(drive_file_id: str):
    """
    Check the processing status of a video.
    
    Returns:
        Current status of the video processing
    """
    
    if drive_file_id not in video_processing_queue:
        raise HTTPException(status_code=404, detail=f"Video {drive_file_id} not found in queue")
    
    return video_processing_queue[drive_file_id]


@router.get("/video-processing-status")
def get_all_video_statuses():
    """
    Get processing status of all videos in queue.
    
    Returns:
        Dictionary with all video processing statuses
    """
    
    return {
        "total_videos": len(video_processing_queue),
        "videos": video_processing_queue
    }


# Import cv2 at the top for frame processing
import cv2