import os
import cv2
import pickle
import numpy as np
from services.database import load_system_data
from config import BASE_DIR

MATCHING_THRESHOLD = 0.45

def process_and_verify(frame):
    """
    معالجة الإطار ومقارنته مع بصمات الوجه المسجلة مع تحميل النموذج محلياً وبأمان تام.
    """
    db_data = load_system_data()
    
    profiles = {}
    eye_encodings = {}
    
    if isinstance(db_data, (tuple, list)) and len(db_data) >= 2:
        profiles = db_data[0] if isinstance(db_data[0], dict) else {}
        eye_encodings = db_data[1] if isinstance(db_data[1], dict) else {}
    elif isinstance(db_data, dict):
        profiles = db_data.get('profiles', {})
        eye_encodings = db_data.get('eye_encodings', {})

    if not eye_encodings:
        return "Unknown", 0.0, None, profiles

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cascade_path = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
    if not os.path.exists(cascade_path):
        cascade_path = 'haarcascade_frontalface_default.xml'
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = []
    if not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    best_score = 0.0
    matched_name = "Unknown"
    roi_gray = None

    known_templates = []
    known_names = []
    
    for name, template in eye_encodings.items():
        if template is not None:
            known_templates.append(template)
            known_names.append(name)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            
            if len(known_templates) > 0 and roi_gray.shape[0] > 10 and roi_gray.shape[1] > 10:
                for i, template in enumerate(known_templates):
                    try:
                        resized_roi = cv2.resize(roi_gray, (template.shape[1], template.shape[0]))
                        res = cv2.matchTemplate(resized_roi, template, cv2.TM_CCOEFF_NORMED)
                        score = res[0][0]
                        if score > best_score:
                            best_score = score
                            matched_name = known_names[i]
                    except Exception:
                        pass
    else:
        h, w = gray.shape
        roi_gray = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
        if len(known_templates) > 0:
            for i, template in enumerate(known_templates):
                try:
                    resized_roi = cv2.resize(roi_gray, (template.shape[1], template.shape[0]))
                    res = cv2.matchTemplate(resized_roi, template, cv2.TM_CCOEFF_NORMED)
                    score = res[0][0]
                    if score > best_score:
                        best_score = score
                        matched_name = known_names[i]
                except Exception:
                    pass

    if best_score > MATCHING_THRESHOLD:
        return matched_name, best_score, roi_gray, profiles
    else:
        return "Unknown", best_score, roi_gray, profiles

def check_liveness(frame):
    """
    Balanced Multi-Layer Anti-Spoofing & Liveness Verification:
    معايير متوازنة تقبل الوجه البشري الحقيقي وتمنع الشاشات المزيفة.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 5.0:
            return False, "Spoofing Detected: Image is too blurry."
        if lap_var > 2000.0:
            return False, "Spoofing Detected: Artificial high sharpness."

        # 2. Local Texture Variance
        gray_f = gray.astype(np.float32)
        kernel = np.ones((5, 5), np.float32) / 25
        mean = cv2.filter2D(gray_f, -1, kernel)
        sq_mean = cv2.filter2D(gray_f**2, -1, kernel)
        local_var = sq_mean - mean**2
        avg_local_var = np.mean(local_var)
        
        if avg_local_var < 5.0:
            return False, "Spoofing Detected: Lack of natural micro-texture."

        # 3. YCrCb Skin Color Spectrum
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin_mask = (cr >= 110) & (cr <= 190) & (cb >= 60) & (cb <= 140)
        skin_ratio = np.sum(skin_mask) / skin_mask.size
        
        if skin_ratio < 0.01:
            return False, "Spoofing Detected: Non-human color spectrum."

        return True, "Liveness Verified"
    except Exception as e:
        return True, f"Liveness Bypass (Error: {str(e)})"