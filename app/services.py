import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
import uuid
import os
import re

class LicensePlateRecognitionService:
    def __init__(self):
        print("📥 Loading AI models... (Please wait)")
        self.model_yolo = YOLO('yolov8n.pt')
        
        # Try to load license plate detection model
        # If not found, we'll use contour detection as fallback
        self.plate_model = None
        try:
            # Attempt to load a pre-trained license plate detection model
            # Using a small YOLO model trained on license plates
            print("📥 Attempting to load license plate detection model...")
            self.plate_model = YOLO('license-plate-detector.pt')
            print("✅ License plate model loaded successfully!")
        except Exception as e:
            print(f"⚠️  License plate model not found: {e}")
            print("💡 Will use contour detection as fallback.")
            self.plate_model = None
        
        self.reader = easyocr.Reader(['en'], gpu=False) 
        print("✅ Models loaded successfully!")

    def detect_license_plate(self, img: np.ndarray) -> tuple:
        """
        Detects license plate using pre-trained model if available.
        Returns (x, y, w, h) of the plate bounding box or None.
        Falls back to contour detection if model is unavailable.
        """
        # Try using the pre-trained plate detection model
        if self.plate_model is not None:
            try:
                results = self.plate_model(img, verbose=False)
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    
                    if len(boxes) > 0:
                        # Get the box with highest confidence
                        best_box = None
                        best_conf = 0
                        
                        for box in boxes:
                            conf = box.conf[0]
                            if conf > best_conf:
                                best_conf = conf
                                best_box = box
                        
                        if best_box is not None:
                            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                            w = x2 - x1
                            h = y2 - y1
                            print(f"   ✅ License plate detected by model: ({x1}, {y1}, {w}, {h}) | Confidence: {best_conf:.2f}")
                            return (x1, y1, w, h)
            except Exception as e:
                print(f"   ⚠️  Plate model detection failed: {e}. Using fallback method...")
        
        # Fallback: Use contour detection
        print(f"   📍 Using contour detection fallback...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            if area > 500 and w > h and 1.5 < (w / h) < 6:
                valid_contours.append((x, y, w, h, area))
        
        if valid_contours:
            valid_contours.sort(key=lambda x: x[4], reverse=True)
            x, y, w, h, _ = valid_contours[0]
            print(f"   ✅ License plate detected by contour: ({x}, {y}, {w}, {h})")
            return (x, y, w, h)
        
        print(f"   ❌ No license plate detected by any method.")
        return None

    def apply_filters(self, img: np.ndarray) -> list:
        """
        Generates 5 different versions of the image to maximize OCR chances.
        Returns a list of tuples: (description, processed_image)
        """
        filters = []

        # 1. Base Pre-processing (Grayscale + Zoom)
        # Upscale 3x to simulate 'Zoom'
        img_zoomed = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img_zoomed, cv2.COLOR_BGR2GRAY)

        # --- OPTION 1: Simple Adaptive Threshold ---
        # Good for general cases
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 5)
        filters.append(("Simple Threshold", thresh1))

        # --- OPTION 2: High Contrast + Dilation (Thicker Letters) ---
        # Good when letters are too thin or broken
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast = clahe.apply(gray)
        binary2 = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary2, kernel, iterations=1)
        filters.append(("Thick Letters", dilated))

        # --- OPTION 3: Gaussian Blur + Otsu (Noise Removal) ---
        # Good for grainy/noisy images
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        filters.append(("De-Noised Otsu", otsu))

        # --- OPTION 4: Negative (Inverted) ---
        # Sometimes OCR prefers white text on black background
        inverted = cv2.bitwise_not(thresh1)
        filters.append(("Inverted Colors", inverted))

        # --- OPTION 5: Gamma Correction (Dark/Bright adjustment) ---
        # Good for shadows
        gamma = 1.5 # Brighten up
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        bright = cv2.LUT(gray, table)
        thresh5 = cv2.adaptiveThreshold(bright, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 5)
        filters.append(("Gamma Corrected", thresh5))

        return filters

    def calculate_score(self, text: str, confidence: float) -> float:
        """
        Scoring System:
        - Base score: Confidence (0 to 1)
        - Bonus: Matches Regex Pattern (+100)
        - Bonus: Correct Length (+10)
        """
        score = confidence
        text_clean = text.upper().replace("-", "").replace(" ", "")

        # Regex validation (Old Brazil & Mercosul)
        pattern_combined = r'[A-Z]{3}[0-9][A-Z0-9][0-9]{2}'
        
        if len(text_clean) == 7:
            score += 10 # Points for correct length
        
        if re.match(pattern_combined, text_clean):
            score += 100 # Jackpot! High points for matching the pattern
            
        return score

    def process_infraction(self, image_bytes: bytes, decibels: float) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result_data = {
            "license_plate": "UNKNOWN",
            "status": "No motorcycle detected",
            "file_path": "",
            "confidence_score": 0.0,
            "needs_verification": False,
            "verification_reason": "",
            "debug_info": []
        }

        results = self.model_yolo(original_img, verbose=False)
        motorcycle_detected = False

        for r in results:
            for box in r.boxes:
                if int(box.cls) == 3:  # Motorcycle
                    motorcycle_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop Strategy: Bottom 50%
                    height = y2 - y1
                    start_y = y1 + int(height * 0.5) 
                    roi_moto = original_img[start_y:y2, x1:x2]
                    
                    if roi_moto.size == 0: continue
                    
                    # Detect license plate within the motorcycle ROI
                    print(f"🔍 Detecting license plate in motorcycle region...")
                    plate_coords = self.detect_license_plate(roi_moto)
                    
                    # Extract the plate ROI (either detected or full)
                    if plate_coords is not None:
                        plate_x, plate_y, plate_w, plate_h = plate_coords
                        # Ensure coordinates are within bounds
                        plate_x = max(0, plate_x)
                        plate_y = max(0, plate_y)
                        plate_x2 = min(roi_moto.shape[1], plate_x + plate_w)
                        plate_y2 = min(roi_moto.shape[0], plate_y + plate_h)
                        roi_plate = roi_moto[plate_y:plate_y2, plate_x:plate_x2]
                    else:
                        # Fallback: use full ROI if detection fails
                        print(f"   💡 Using full motorcycle ROI as fallback")
                        roi_plate = roi_moto
                    
                    if roi_plate.size == 0:
                        print("   ❌ Plate ROI is invalid. Skipping...")
                        continue
                    
                    # 1. Generate 5 Candidates - using the plate region only
                    candidates = self.apply_filters(roi_plate)
                    
                    best_text = "UNKNOWN"
                    best_score = -1
                    best_filter_name = ""

                    print(f"--- 🏁 STARTING OCR TOURNAMENT ---")

                    # 2. Run OCR on all candidates
                    for i, (filter_name, processed_img) in enumerate(candidates):
                        # Save debug image
                        cv2.imwrite(f"static/debug_filter_{i}.jpg", processed_img)

                        readings = self.reader.readtext(processed_img, detail=1, paragraph=False)
                        
                        # Concatenate and clean
                        full_text = "".join([res[1] for res in readings]).upper().replace(" ", "").replace("-", "")
                        avg_conf = np.mean([res[2] for res in readings]) if readings else 0.0
                        
                        # 3. Score the result
                        score = self.calculate_score(full_text, avg_conf)
                        
                        print(f"Filter [{filter_name}]: Read '{full_text}' | Score: {score:.2f}")

                        if score > best_score:
                            best_score = score
                            best_text = full_text
                            best_filter_name = filter_name

                    print(f"🏆 WINNER: {best_filter_name} with '{best_text}'")
                    
                    # Normalize score to 0-100 scale
                    # Max possible: 1.0 (confidence) + 10 (length) + 100 (pattern) = 111
                    normalized_score = min((best_score / 111.0) * 100, 100.0)
                    result_data['confidence_score'] = normalized_score
                    
                    print(f"📊 Normalized Confidence Score: {normalized_score:.1f}%")
                    
                    # Check if score is below 85% - FLAG FOR MANUAL VERIFICATION
                    if normalized_score < 85:
                        result_data['needs_verification'] = True
                        result_data['verification_reason'] = f"Low confidence score ({normalized_score:.1f}%). Manual verification required to ensure accuracy."
                        print(f"⚠️  LOW CONFIDENCE WARNING ⚠️")
                        print(f"   Score: {normalized_score:.1f}% (below 85% threshold)")
                        print(f"   Action: MANUAL VERIFICATION REQUIRED")
                        print(f"   Please have someone review this image to ensure the detected characters are correct.")
                    else:
                        print(f"✅ High confidence ({normalized_score:.1f}%) - No additional verification needed")

                    # Post-Process Winner (Regex Extraction)
                    # Try to extract exactly the 7 chars
                    match = re.search(r'[A-Z]{3}[0-9][A-Z0-9][0-9]{2}', best_text)
                    if match:
                        final_plate = match.group()
                        result_data['license_plate'] = final_plate
                        
                        # Draw on original
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Color code: Green for high confidence, Yellow for low confidence
                        text_color = (0, 255, 0) if not result_data['needs_verification'] else (0, 255, 255)
                        confidence_label = f"({normalized_score:.1f}%)"
                        label = f"PLATE: {final_plate} {confidence_label}"
                        cv2.putText(original_img, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                        break 
                    else:
                        # Pattern not found - still flag as needing verification
                        result_data['needs_verification'] = True
                        if not result_data['verification_reason']:
                            result_data['verification_reason'] = "Could not match exact license plate pattern despite processing. Manual verification required." 

            if result_data['license_plate'] != "UNKNOWN":
                break

        filename = f"{uuid.uuid4()}.jpg"
        save_path = os.path.join("static", "images", filename)
        cv2.imwrite(save_path, original_img)
        
        result_data['file_path'] = f"/static/images/{filename}"
        if motorcycle_detected:
            result_data['status'] = "Processed"
        
        return result_data