import os
import pickle
import json
from config import BASE_DIR

PROFILES_FILE = os.path.join(BASE_DIR, 'profiles.json')
ENCODINGS_FILE = os.path.join(BASE_DIR, 'eye_encodings.pickle')

def load_system_data():
    """
    تحميل الملفات الشخصية وبصمات الوجوه من التخزين المحلي (JSON و Pickle).
    """
    profiles = {}
    eye_encodings = {}
    
    try:
        if os.path.exists(PROFILES_FILE):
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
    except Exception as e:
        profiles = {}

    try:
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, 'rb') as f:
                eye_encodings = pickle.load(f)
    except Exception as e:
        eye_encodings = {}
        
    return profiles, eye_encodings

def save_user_profile(key_name, profile_data, roi):
    """
    حفظ ملف مستخدم جديد مع بصمة الوجه (ROI) في النظام.
    """
    profiles, eye_encodings = load_system_data()
    
    profiles[key_name] = profile_data
    eye_encodings[key_name] = roi
    
    try:
        with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=4)
    except Exception as e:
        pass

    try:
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(eye_encodings, f)
    except Exception as e:
        pass