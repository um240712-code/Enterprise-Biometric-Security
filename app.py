import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from services.biometric import process_and_verify, check_liveness
from services.database import save_user_profile, load_system_data
from config import INTRUDERS_DIR, LOG_FILE, BASE_DIR

app = Flask(__name__)

# ذاكرة مؤقتة لحفظ بصمة الوجه الحالية لضمان نجاح التسجيل الفوري
CURRENT_SESSION = {
    "last_roi": None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            pass
            
    log_data = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            log_data = f.readlines()
            
    intruders = []
    if os.path.exists(INTRUDERS_DIR):
        intruders = os.listdir(INTRUDERS_DIR)
        
    return render_template('admin.html', logs=log_data[::-1], intruders=intruders)

@app.route('/api/verify', methods=['POST'])
def verify():
    try:
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Invalid image frame'})

        # فحص الحيوية أولاً (Anti-Spoofing)
        is_live, liveness_msg = check_liveness(frame)
        
        if not is_live:
            filename = f"spoof_{os.urandom(4).hex()}.jpg"
            if not os.path.exists(INTRUDERS_DIR):
                os.makedirs(INTRUDERS_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(INTRUDERS_DIR, filename), frame)
            
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[SECURITY ALERT] {liveness_msg} - Saved as {filename}\n")
                
            return jsonify({
                'status': 'intruder',
                'message': liveness_msg,
                'score': 0.0
            })

        # مطابقة البصمة الحيوية
        matched_name, score, roi_gray, profiles = process_and_verify(frame)
        
        if roi_gray is not None and roi_gray.shape[0] > 10:
            CURRENT_SESSION["last_roi"] = roi_gray

        # تحويل الـ score إلى نوع float عادي لتجنب مشكلة الـ JSON
        py_score = float(score) if score is not None else 0.0

        if matched_name != "Unknown":
            user_profile = profiles.get(matched_name, {})
            with open(LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(f"[ACCESS GRANTED] User: {matched_name} [Score: {py_score:.2f}]\n")
            return jsonify({
                'status': 'success',
                'profile': user_profile,
                'score': py_score
            })
        else:
            return jsonify({
                'status': 'intruder',
                'message': 'Unauthorized entity. Face template not recognized.',
                'score': py_score
            })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/register', methods=['POST'])
def register():
    try:
        key_name = request.form.get('key_name')
        if not key_name:
            return jsonify({'status': 'error', 'message': 'Username key is required'})
            
        roi = CURRENT_SESSION.get("last_roi")
        if roi is None:
            return jsonify({'status': 'error', 'message': 'No valid face template captured. Please capture your face first.'})
            
        profile_data = {
            "full_name": request.form.get('full_name'),
            "id_number": request.form.get('id_number'),
            "phone_number": request.form.get('phone_number'),
            "major": request.form.get('major'),
            "institution": request.form.get('institution'),
            "location": request.form.get('location'),
            "education_path": request.form.get('education_path'),
            "family_info": {
                "father_status": request.form.get('f_status'),
                "father_age": request.form.get('f_age'),
                "father_occupation": request.form.get('f_occ'),
                "mother_status": request.form.get('m_status'),
                "mother_age": request.form.get('m_age'),
                "mother_occupation": request.form.get('m_occ')
            },
            "technical_profile": request.form.get('tech_profile'),
            "clearance_level": request.form.get('clearance', 'User')
        }
        
        save_user_profile(key_name, profile_data, roi)
        CURRENT_SESSION["last_roi"] = None
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/intruders/<filename>')
def get_intruder_image(filename):
    return send_from_directory(INTRUDERS_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)