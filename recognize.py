import cv2
import pickle
import numpy as np
import os
import json
from datetime import datetime

# إنشاء المجلدات الضرورية
os.makedirs("intruders", exist_ok=True)
os.makedirs("verified_snapshots", exist_ok=True)
os.makedirs("known_faces", exist_ok=True)

if not os.path.exists("profiles.json"):
    with open("profiles.json", "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=4)

def load_system_data():
    if os.path.exists("eye_encodings.pickle"):
        with open("eye_encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
        return data.get("features", []), data.get("names", [])
    return [], []

known_templates, known_names = load_system_data()

with open("profiles.json", "r", encoding="utf-8") as f:
    profiles = json.load(f)

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def print_profile_dashboard(p):
    fam = p.get("family_info", {})
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "╔" + "═"*65 + "╗")
    print(f"║ 🛡️  [ENTERPRISE SECURITY DASHBOARD] - تقرير التحقق الأمني       ║")
    print("╠" + "═"*65 + "╣")
    print(f"║ 👤 الاسم الكامل    : {p.get('full_name')}      ║")
    print(f"║ 🆔 رقم الهوية      : {p.get('id_number')}                         ║")
    print(f"║ 📱 رقم الهاتف      : {p.get('phone_number')}                  ║")
    print(f"║ 🎓 التخصص الأكاديمي: {p.get('major')} ║")
    print(f"║ 🏛️  الكلية/المؤسسة : {p.get('institution')}      ║")
    print(f"║ 📍 السكن الحالي    : {p.get('location')}                 ║")
    print(f"║ 📚 المسار التعليمي : {p.get('education_path')}               ║")
    print("╟" + "─"*65 + "┢")
    print(f"║ 父亲 حالة الأب     : {fam.get('father_status')} | العمر: {fam.get('father_age')} | العمل: {fam.get('father_occupation')} ║")
    print(f"║ 母亲 حالة الأم     : {fam.get('mother_status')} | العمر: {fam.get('mother_age')} | العمل: {fam.get('mother_occupation')} ║")
    print("╟" + "─"*65 + "┢")
    print(f"║ 💻 المهارات والتقنيات: {p.get('technical_profile')}       ║")
    print(f"║ 🔒 مستوى الصلاحية  : {p.get('clearance_level')}          ║")
    print(f"║ ⏱️  وقت التحقق     : {current_time}                ║")
    print("╚" + "═"*65 + "╝\n")

# ==================== القائمة الرئيسية ====================
print("\n" + "="*55)
print("🛡️ نظام إدارة الهوية والتعرف الذكي - القائمة الرئيسية")
print("="*55)
print("1. وضع العرض الحي المستمر (Live Stream Mode)")
print("2. وضع التقاط صورة فردية والتحقق مع التسجيل الديناميكي")
print("="*55)
choice = input("اختر وضع التشغيل (أدخل 1 أو 2): ").strip()

cap = cv2.VideoCapture(0)

if choice == "1":
    print("\n[INFO] تم تفعيل [وضع العرض الحي المستمر]... (اضغط 'q' للخروج)")
    from collections import deque, Counter
    prediction_history = deque(maxlen=5)
    last_logged_name = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        current_frame_name = "Unknown"
        best_score = 0
        x, y, w, h = 0, 0, 0, 0

        if len(eyes) >= 1:
            min_x = min([e[0] for e in eyes])
            min_y = min([e[1] for e in eyes])
            max_x = max([e[0] + e[2] for e in eyes])
            
            x = max(0, min_x - 45)
            y = max(0, min_y - 50)
            w = (max_x - min_x) + 90
            h = 140
            
            roi_gray = gray[y:y+h, x:x+w]
            
            if len(known_templates) > 0 and roi_gray.shape[0] > 10 and roi_gray.shape[1] > 10:
                for i, template in enumerate(known_templates):
                    try:
                        resized_roi = cv2.resize(roi_gray, (template.shape[1], template.shape[0]))
                        res = cv2.matchTemplate(resized_roi, template, cv2.TM_CCOEFF_NORMED)
                        score = res[0][0]
                        if score > best_score:
                            best_score = score
                            best_name = known_names[i]
                    except:
                        pass
                
                if best_score > 0.13:
                    current_frame_name = best_name

            prediction_history.append(current_frame_name)
            counts = Counter(prediction_history)
            most_common_name, frequency = counts.most_common(1)[0]
            stable_name = most_common_name if (most_common_name != "Unknown" and frequency >= 2) else "Unknown"

            if stable_name != "Unknown" and stable_name in profiles:
                if last_logged_name != stable_name:
                    print_profile_dashboard(profiles[stable_name])
                    last_logged_name = stable_name
            else:
                if stable_name == "Unknown":
                    last_logged_name = None

            color = (0, 255, 0) if stable_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{stable_name} ({best_score:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Live Stream Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

elif choice == "2":
    print("\n[INFO] تم تفعيل [وضع التقاط صورة فردية والتحقق]...")
    print("[INSTRUCTION] انظر للكاميرا واضغط مفتاح [Space / المسافة] لالتقاط الصورة والتحقق، أو 'q' للخروج.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        cv2.putText(display_frame, "Press [SPACE] to Capture & Verify", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Snapshot Verification Mode", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # عند الضغط على مسافة
            print("[INFO] جاري التقاط الصورة وتحليل بصمة العين...")
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

            matched_name = "Unknown"
            best_score = 0
            roi_gray = None

            if len(eyes) >= 1:
                min_x = min([e[0] for e in eyes])
                min_y = min([e[1] for e in eyes])
                max_x = max([e[0] + e[2] for e in eyes])
                
                x = max(0, min_x - 45)
                y = max(0, min_y - 50)
                w = (max_x - min_x) + 90
                h = 140
                
                roi_gray = gray[y:y+h, x:x+w]
                
                if len(known_templates) > 0 and roi_gray.shape[0] > 10 and roi_gray.shape[1] > 10:
                    for i, template in enumerate(known_templates):
                        try:
                            resized_roi = cv2.resize(roi_gray, (template.shape[1], template.shape[0]))
                            res = cv2.matchTemplate(resized_roi, template, cv2.TM_CCOEFF_NORMED)
                            score = res[0][0]
                            if score > best_score:
                                best_score = score
                                best_name = known_names[i]
                        except:
                            pass
                    
                    if best_score > 0.13:
                        matched_name = best_name

            if matched_name != "Unknown" and matched_name in profiles:
                print(f"\n[SUCCESS] تم التحقق بنجاح! المطابقة بنسبة: ({best_score:.2f})")
                print_profile_dashboard(profiles[matched_name])
                snapshot_path = f"verified_snapshots/verified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(snapshot_path, frame)
                print(f"💾 تم حفظ لقطة التحقق في: {snapshot_path}\n")
            else:
                print(f"\n[!] تحذير أمني: لم يتم التعرف على الهوية (Unknown - Score: {best_score:.2f})")
                intruder_path = f"intruders/intruder_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(intruder_path, frame)
                print(f"🚨 تم التقاط صورة المتسلل وحفظها في: {intruder_path}")
                
                register_choice = input("هل تريد تسجيل هذا الشخص وإدخال بياناته الآن؟ (y للتسجيل / n للتخطي): ").strip().lower()
                if register_choice == 'y':
                    if roi_gray is not None and roi_gray.shape[0] > 10:
                        print("\n" + "╔" + "═"*50 + "╗")
                        print("║        📝 نموذج تسجيل بيانات مستخدم جديد          ║")
                        print("╚" + "═"*50 + "╝")
                        
                        # عرض قائمة الحقول المطلوبة بشكل منظم
                        new_key_name = input("1. المعرف بالإنجليزية (مثل: ahmad): ").strip()
                        full_name = input("2. الاسم الكامل: ").strip()
                        id_num = input("3. رقم الهوية: ").strip()
                        phone = input("4. رقم الهاتف: ").strip()
                        major = input("5. التخصص الأكاديمي: ").strip()
                        institution = input("6. الكلية / المؤسسة: ").strip()
                        location = input("7. مكان السكن: ").strip()
                        edu_path = input("8. المسار التعليمي: ").strip()
                        
                        print("\n--- [ معلومات العائلة ] ---")
                        f_status = input("9. حالة الأب: ").strip()
                        f_age = input("10. عمر الأب: ").strip()
                        f_occ = input("11. عمل الأب: ").strip()
                        m_status = input("12. حالة الأم: ").strip()
                        m_age = input("13. عمر الأم: ").strip()
                        m_occ = input("14. عمل الأم: ").strip()
                        
                        tech_profile = input("15. المهارات التقنية: ").strip()
                        clearance = input("16. مستوى الصلاحية (Admin / User): ").strip()

                        # حفظ بصمة العين الجديدة
                        standard_template = cv2.resize(roi_gray, (100, 50))
                        known_templates.append(standard_template)
                        known_names.append(new_key_name)

                        data_to_save = {"features": known_templates, "names": known_names}
                        with open("eye_encodings.pickle", "wb") as f:
                            f.write(pickle.dumps(data_to_save))

                        # تحديث ملف profiles.json
                        profiles[new_key_name] = {
                            "full_name": full_name,
                            "id_number": id_num,
                            "phone_number": phone,
                            "major": major,
                            "institution": institution,
                            "location": location,
                            "education_path": edu_path,
                            "family_info": {
                                "father_status": f_status,
                                "father_age": f_age,
                                "father_occupation": f_occ,
                                "mother_status": m_status,
                                "mother_age": m_age,
                                "mother_occupation": m_occ
                            },
                            "technical_profile": tech_profile,
                            "clearance_level": clearance,
                            "system_status": "مصرح بالدخول الفوري"
                        }

                        with open("profiles.json", "w", encoding="utf-8") as f:
                            json.dump(profiles, f, ensure_ascii=False, indent=4)

                        print(f"\n[SUCCESS] تم تسجيل المستخدم '{full_name}' وحفظ بياناته في نموذج النظام بنجاح تام!\n")
                    else:
                        print("[!] تعذر استخراج بصمة العين بوضوح من اللقطة، حاول مرة أخرى.\n")

else:
    print("[!] اختيار غير صحيح، يرجى إعادة تشغيل البرنامج واختيار 1 أو 2.")

cap.release()
cv2.destroyAllWindows()