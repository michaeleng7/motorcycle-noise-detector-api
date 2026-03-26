import requests
from typing import Optional
from datetime import datetime
import asyncio

class BackendIntegrationService:
    """
    Service to handle communication with the backend colleague.
    Sends detected plate codes to the backend API.
    """
    
    def __init__(self, backend_url: str = "http://localhost:5000"):
        """
        Initialize backend integration service.
        
        Args:
            backend_url: Base URL of the colleague's backend API
                        Example: "http://localhost:5000" or "https://api.backend.com"
        """
        self.backend_url = backend_url.rstrip('/')
        self.timeout = 10  # seconds
        self.max_retries = 3
        self.retry_delay = 2  # seconds
    
    async def send_plate_code_to_backend(
        self, 
        plate_code: str,
        confidence: float,
        video_id: str,
        additional_info: Optional[dict] = None
    ) -> dict:
        """
        Send detected plate code to the backend colleague's API.
        
        Args:
            plate_code: The recognized license plate code (e.g., "ABC1234")
            confidence: Confidence score (0.0 to 1.0)
            video_id: Google Drive file ID of the processed video
            additional_info: Optional additional data (e.g., decibels, timestamp, etc.)
            
        Returns:
            Response from backend API or error status
        """
        
        payload = {
            "plate_code": plate_code,
            "confidence": confidence,
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "additional_info": additional_info or {}
        }
        
        url = f"{self.backend_url}/plate-code"
        
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"📤 Sending plate code to backend (attempt {attempt}/{self.max_retries})")
                print(f"   URL: {url}")
                print(f"   Plate: {plate_code} | Confidence: {confidence:.2%}")
                
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    print(f"✅ Plate code sent successfully to backend!")
                    return {
                        "status": "success",
                        "message": "Plate code delivered to backend",
                        "backend_response": response.json() if response.text else {}
                    }
                else:
                    print(f"⚠️  Backend returned status {response.status_code}")
                    print(f"   Response: {response.text}")
                    
                    if attempt < self.max_retries:
                        print(f"   ⏳ Retrying in {self.retry_delay}s...")
                        await asyncio.sleep(self.retry_delay)
                    continue
                    
            except requests.exceptions.Timeout:
                print(f"⏱️  Request timeout ({self.timeout}s)")
                if attempt < self.max_retries:
                    print(f"   ⏳ Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    
            except requests.exceptions.ConnectionError:
                print(f"❌ Connection error - Backend at {url} is unreachable")
                if attempt < self.max_retries:
                    print(f"   ⏳ Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
                    
            except Exception as e:
                print(f"❌ Error sending plate code: {e}")
                if attempt < self.max_retries:
                    print(f"   ⏳ Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
        
        # All retries failed
        return {
            "status": "error",
            "message": f"Failed to send plate code after {self.max_retries} attempts",
            "plate_code": plate_code,
            "backend_url": url
        }
    
    async def notify_video_received(self, video_id: str) -> dict:
        """
        Notify backend that a video has been received and processing started.
        
        Args:
            video_id: Google Drive file ID
            
        Returns:
            Acknowledgment from backend
        """
        
        payload = {
            "video_id": video_id,
            "status": "processing_started",
            "timestamp": datetime.now().isoformat()
        }
        
        url = f"{self.backend_url}/video-processing-status"
        
        try:
            print(f"📢 Notifying backend of video processing start: {video_id}")
            
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ Backend acknowledged video processing notification")
                return {"status": "acknowledged"}
            else:
                print(f"⚠️  Backend notification returned status {response.status_code}")
                return {"status": "warning", "code": response.status_code}
                
        except Exception as e:
            print(f"⚠️  Could not notify backend: {e}")
            return {"status": "notification_skipped", "reason": str(e)}
