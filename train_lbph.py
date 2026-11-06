import cv2
import os
import json
import numpy as np
import tempfile

def load_cascade_safely(filename):
    """解決 Windows 中文路徑導致 OpenCV 無法載入 XML 的函式"""
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

def imread_unicode(file_path):
    """解決 cv2.imread 無法讀取中文路徑圖片的問題"""
    try:
        with open(file_path, 'rb') as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            return cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

face_cascade = load_cascade_safely('haarcascade_frontalface_alt.xml')
if face_cascade is None:
    print("[錯誤] 無法載入 Haar Cascade 模型！")
    exit()

face_samples = []
labels = []
label_map = {}
current_id = 0

print("讀取人臉資料集中...")

if not os.path.exists(data_dir):
    print("錯誤：未找到 data 資料夾，請先執行 collect_faces.py 收集照片！")
    exit()

# 直接遍歷 data/ 目錄下的所有子資料夾（只讀第一層）
subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]

for dir_name in subfolders:
    person_dir = os.path.join(data_dir, dir_name)
    if not os.path.exists(person_dir):
        continue

    # 取得該人物資料夾內的所有圖片
    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) > 0:
        label_map[current_id] = dir_name
        for file_name in image_files:
            img_path = os.path.join(person_dir, file_name)
            img = imread_unicode(img_path)
            if img is None:
                continue

            # 檢測與對齊人臉
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = img[y:y+h, x:x+w]
            else:
                face_roi = img

            face_roi = cv2.resize(face_roi, (200, 200))
            face_samples.append(face_roi)
            labels.append(current_id)

        print(f"  - 已載入 [{dir_name}]: {len(image_files)} 張照片")
        current_id += 1

if len(face_samples) == 0:
    print("錯誤：未找到可用的訓練影像，請先執行 collect_faces.py 收集照片！")
    exit()

print(f"\n共找到 {len(label_map)} 位對象，總計 {len(face_samples)} 張人臉圖像，開始訓練 LBPH 模型...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_samples, np.array(labels))

# 使用 Temp 目錄繞過 LBPH recognizer.save 在 Windows 中文路徑下的問題
temp_model_path = os.path.join(tempfile.gettempdir(), "lbph_model.yml")
recognizer.save(temp_model_path)

model_path = os.path.join(models_dir, "lbph_model.yml")
labels_path = os.path.join(models_dir, "labels.json")

# 將寫好的模型複製回專案目錄
with open(temp_model_path, "rb") as f_src:
    with open(model_path, "wb") as f_dst:
        f_dst.write(f_src.read())

with open(labels_path, "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print(f"\n訓練完畢！模型已成功儲存至:\n  - {model_path}\n  - {labels_path}")