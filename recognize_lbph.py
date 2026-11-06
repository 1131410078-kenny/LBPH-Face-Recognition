import cv2
import json
import os
import tempfile

def load_cascade_safely(filename):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    local_xml = os.path.join(BASE_DIR, filename)
    target_path = local_xml if os.path.exists(local_xml) else os.path.join(cv2.data.haarcascades, filename)
    
    if not os.path.exists(target_path):
        return None

    try:
        with open(target_path, 'rb') as f:
            content = f.read()
        
        temp_dir = tempfile.gettempdir()
        temp_xml_path = os.path.join(temp_dir, filename)
        with open(temp_xml_path, 'wb') as f:
            f.write(content)
            
        cascade = cv2.CascadeClassifier(temp_xml_path)
        return cascade if not cascade.empty() else None
    except Exception:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "lbph_model.yml")
labels_path = os.path.join(BASE_DIR, "models", "labels.json")

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print("錯誤：未找到模型或標籤對應檔，請先執行 train_lbph.py！")
    exit()

face_cascade = load_cascade_safely('haarcascade_frontalface_alt.xml')
if face_cascade is None:
    print("[錯誤] 無法載入 Haar Cascade 模型！")
    exit()

with open(labels_path, "r", encoding="utf-8") as f:
    label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

# 透過 Temp 目錄載入 LBPH 模型，解決中文路徑問題
temp_model_path = os.path.join(tempfile.gettempdir(), "lbph_model_load.yml")
with open(model_path, "rb") as f_src:
    with open(temp_model_path, "wb") as f_dst:
        f_dst.write(f_src.read())

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(temp_model_path)

THRESHOLD = 75.0

cap = cv2.VideoCapture(0)
print("開始即時人臉辨識，按下 'q' 鍵退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法取得 Webcam 影像！")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        label_id, confidence = recognizer.predict(face_roi)

        if confidence <= THRESHOLD and label_id in label_map:
            name = label_map[label_id]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, f"{name} ({confidence:.1f})", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    cv2.imshow("LBPH Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()