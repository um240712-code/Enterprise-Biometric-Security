import cv2
import pickle
import os
import urllib.request

face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye.xml'

if not os.path.exists(face_cascade_path):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml', face_cascade_path)
if not os.path.exists(eye_cascade_path):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml', eye_cascade_path)

eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

dataset_dir = "known_faces"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

print("[INFO] جاري تشغيل الكاميرا لالتقاط صورتك...")
print("[INSTRUCTION] ارتدِ القناع (أو انظر للكاميرا) واضغط حرف (s) لتسجيل بصمة العين بدقة!")

cap = cv2.VideoCapture(0)
name = "jibreil"
saved = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Enrollment - Press 's' to capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        image_path = os.path.join(dataset_dir, f"{name}.jpg")
        cv2.imwrite(image_path, frame)
        saved = True
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if saved:
    image = cv2.imread(os.path.join(dataset_dir, f"{name}.jpg"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)  # تطبيق نفس معالجة الإضاءة تماماً
    
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    known_features = []
    known_names = []
    
    if len(eyes) >= 1:
        min_x = min([e[0] for e in eyes])
        min_y = min([e[1] for e in eyes])
        max_x = max([e[0] + e[2] for e in eyes])
        
        x = max(0, min_x - 35)
        y = max(0, min_y - 45)
        w = (max_x - min_x) + 70
        h = 130
        
        eye_roi = gray[y:y+h, x:x+w]
        if eye_roi.shape[0] > 10 and eye_roi.shape[1] > 10:
            known_features.append(eye_roi)
            known_names.append(name)
            
            data = {"features": known_features, "names": known_names}
            with open("eye_encodings.pickle", "wb") as f:
                f.write(pickle.dumps(data))
            print("[SUCCESS] تم استخراج وحفظ بصمة العين بدقة متناهية!")
        else:
            print("[!] المنطقة المحددة صغيرة جداً، اقترب بوضوح.")
    else:
        print("[!] لم يتم رصد العيون بوضوح في الصورة الملتقطة، حاول مرة أخرى.")