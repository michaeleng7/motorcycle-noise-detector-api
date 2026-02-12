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
        # Using portuguese and english for better OCR accuracy on Brazilian plates
        self.reader = easyocr.Reader(['pt', 'en'], gpu=False) 
        print("✅ Models loaded successfully!")

    def clean_ocr_text(self, text: str) -> str:
        """
        Cleans and corrects confusing characters in OCR.
        """
        # Corrections for common OCR confusions in license plates
        replacements = {
            'O': '0',  # Character O could be confused with zero
            'I': '1',  # Character I could be confused with number 1
            'l': '1',  # Lowercase letter l could be confused with number 1
            'S': '5',  # Character S could be confused with number 5
            'B': '8',  # Character B could be confused with number 8
        }
        
        # Apply replacements to text
        result = ""
        text = text.upper().strip()
        
        for char in text:
            result += replacements.get(char, char)
        
        return result

    def extract_license_plate(self, full_text: str) -> tuple[str, str]:
        """
        Try to extract the license plate from the full text.
        Only accepts strict patterns (MERCOSUL or OLD).
        Returns None if no exact match is found.
        """
        full_text = full_text.upper().replace(" ", "").replace("-", "")
        
        # Strict patterns for Brazilian plates:
        # Mercosul (3 letters, 1 number, 1 letter, 2 numbers) -> Ex: ABC1D23
        pattern_mercosul = r'[A-Z]{3}[0-9][A-Z][0-9]{2}'
        
        # Old plates (3 letters, 4 numbers) -> Ex: ABC1234
        pattern_old = r'[A-Z]{3}[0-9]{4}'
        
        patterns = [
            (pattern_mercosul, "MERCOSUL"),
            (pattern_old, "OLD"),
        ]
        
        for pattern, plate_type in patterns:
            match = re.search(pattern, full_text)
            if match:
                plate = match.group()
                print(f"   🎯 Pattern '{plate_type}' found: {plate}")
                return plate, plate_type
        
        # If no strict pattern found, return None (invalid)
        print(f"   ❌ No valid pattern found in: '{full_text}'")
        return None, None

    def enhance_image_for_ocr(self, roi_img: np.ndarray) -> np.ndarray:
        """
        Applies filters to the ROI to improve OCR accuracy.
        Uses a less aggressive approach to preserve character details.
        """
        # Increasing scale (Zoom) helps OCR see small letters
        img_scaled = cv2.resize(roi_img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE first (preserves details)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrasted = clahe.apply(gray)
        
        # Light noise reduction
        denoised = cv2.fastNlMeansDenoising(contrasted, h=10)
        
        # Adaptive binarization with parameters optimized for license plates
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary

    def ocr_with_confidence_filter(self, image, min_confidence: float = 0.3) -> list:
        """
        Performs OCR and returns list of (text, confidence).
        Does not filter yet - we will do intelligent post-processing.
        """
        results = self.reader.readtext(
            image,
            detail=1,  # Returns details including confidence
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        )
        
        # Returns everything, we will process later
        text_confidence_pairs = [(text.upper(), confidence) for (bbox, text, confidence) in results]
        
        # Detailed log
        print(f"   - Raw details: {text_confidence_pairs}")
        
        return text_confidence_pairs

    def merge_ocr_results(self, text_conf_pairs: list, min_confidence: float = 0.3) -> str:
        """
        Intelligently merges OCR results, considering confidence and license plate logic.
        Corrects common OCR confusions based on position context.
        """
        if not text_conf_pairs:
            return ""
        
        # Filters by minimum confidence (more permissive now)
        valid_pairs = [(text, conf) for text, conf in text_conf_pairs if conf >= min_confidence or len(text) > 1]
        
        if not valid_pairs:
            return ""
        
        # Merge everything
        merged = "".join([text for text, conf in valid_pairs])
        
        # Intelligent post-processing
        merged_upper = merged.upper().replace(" ", "").replace("-", "")
        
        # Strategy 1: If contains "BRASIL", extract the first 3 letters (BRA)
        # and try to merge with next elements to form a plate
        if 'BRASIL' in merged_upper:
            # Remove BRASIL and get the rest
            after_brasil = merged_upper.replace('BRASIL', '')
            # Try: BRA + rest
            merged_upper = 'BRA' + after_brasil
        
        # Strategy 2: Correct common OCR confusions based on context
        # Position 0-2: Must be letters
        # Position 3: Must be number
        # Position 4: Must be letter (Mercosul) or number (Old)
        # Position 5-6: Must be numbers
        
        corrected = ""
        for i, char in enumerate(merged_upper):
            if i < 3:  # First 3 positions must be letters
                if char.isdigit():
                    # Try to convert number to letter
                    corrections = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
                    corrected += corrections.get(char, char)
                else:
                    corrected += char
            elif i == 3:  # Position 3 must be number
                if not char.isdigit():
                    # Try to convert letter to number
                    corrections = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}
                    corrected += corrections.get(char, char)
                else:
                    corrected += char
            elif i == 4:  # Position 4 can be letter (Mercosul) or number (Old)
                # Keep as is, will be validated by pattern
                corrected += char
            else:  # Positions 5+ should be numbers
                if not char.isdigit():
                    corrections = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}
                    corrected += corrections.get(char, char)
                else:
                    corrected += char
        
        merged_upper = corrected if len(corrected) >= 7 else merged_upper
        
        print(f"   📊 Merged result: '{merged_upper}'")
        
        return merged_upper

    def process_infraction(self, image_bytes: bytes, decibels: float) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result_data = {
            "license_plate": "UNKNOWN",
            "status": "No motorcycle detected",
            "file_path": ""
        }

        results = self.model_yolo(original_img, verbose=False)
        motorcycle_detected = False

        for r in results:
            for box in r.boxes:
                if int(box.cls) == 3:  # ID 3 = motorcycle
                    motorcycle_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Crop Strategy: Try different areas of the motorcycle
                    height = y2 - y1
                    
                    # Strategy 1: Bottom 40% (more likely to have front plate)
                    roi_strategies = [
                        ("Bottom 40%", original_img[y1 + int(height * 0.6):y2, x1:x2]),
                        ("Bottom 50%", original_img[y1 + int(height * 0.5):y2, x1:x2]),
                        ("Middle-Bottom", original_img[y1 + int(height * 0.3):y2, x1:x2]),
                        ("Full Motor", original_img[y1:y2, x1:x2]),
                    ]
                    
                    best_plate = None
                    best_type = None
                    best_strategy = None
                    
                    for strategy_name, roi_moto in roi_strategies:
                        if roi_moto.size == 0:
                            continue
                        
                        print(f"\n🔄 Trying strategy: {strategy_name}")
                        
                        # ===== APPROACH 1: WITH PROCESSING =====
                        processed_roi = self.enhance_image_for_ocr(roi_moto)
                        
                        # Save processed ROI for debug (optional, comment if you don't want)
                        # debug_path = os.path.join("static", "images", f"debug_{strategy_name.replace(' ', '_')}.jpg")
                        # cv2.imwrite(debug_path, processed_roi)
                        
                        text_conf_processed = self.ocr_with_confidence_filter(processed_roi)
                        text_processed = self.merge_ocr_results(text_conf_processed, min_confidence=0.5)
                        
                        plate, plate_type = self.extract_license_plate(text_processed)
                        if plate:
                            best_plate = plate
                            best_type = plate_type
                            best_strategy = f"{strategy_name} (processed)"
                            print(f"   ✅ SUCCESS! Plate: {plate} ({plate_type})")
                            break
                        
                        # ===== APPROACH 2: WITHOUT PROCESSING (Raw) =====
                        print(f"   Trying raw (without processing)...")
                        text_conf_raw = self.ocr_with_confidence_filter(roi_moto)
                        text_raw = self.merge_ocr_results(text_conf_raw, min_confidence=0.5)
                        
                        plate, plate_type = self.extract_license_plate(text_raw)
                        if plate:
                            best_plate = plate
                            best_type = plate_type
                            best_strategy = f"{strategy_name} (raw)"
                            print(f"   ✅ SUCCESS! Plate: {plate} ({plate_type})")
                            break
                    
                    if best_plate:
                        result_data['license_plate'] = best_plate
                        
                        # Draw on image
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"PLATE: {best_plate} | {decibels}dB"
                        cv2.putText(original_img, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        print(f"\n🎯 Best result - Strategy: {best_strategy}")
                        break

            if result_data['license_plate'] != "UNKNOWN":
                break

        # Save Image
        filename = f"{uuid.uuid4()}.jpg"
        save_path = os.path.join("static", "images", filename)
        cv2.imwrite(save_path, original_img)
        
        result_data['file_path'] = f"/static/images/{filename}"
        if motorcycle_detected:
            result_data['status'] = "Processed"
        
        return result_data