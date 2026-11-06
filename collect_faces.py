import cv2
import os
import tempfile

def load_cascade_safely(filename):
    """解決 Windows 中文路徑導致 OpenCV 無法載入 XML 的函式"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    local_xml = os.path.join(BASE_DIR, filename)
    
    # 優先嘗試讀取專案目錄下的 XML，若無則讀取 cv2 內建 XML
    target_path = local_xml if os.path.exists(local_xml) else os.path.join(cv2.data.haarcascades, filename)
    
    if not os.path.exists(target_path):
        return None

    try:
        # 使用 Python 原生 binary 方式讀取檔案
        with open(target_path, 'rb') as f:
            content = f.read()
        
        # 寫入無中文路徑的系統 Temp 目錄
        temp_dir = tempfile.gettempdir()
        temp_xml_path = os.path.join(temp_dir, filename)
        with open(temp_xml_path, 'wb') as f:
            f.write(content)
            
        cascade = cv2.CascadeClassifier(temp_xml_path)
        return cascade if not cascade.empty() else None
    except Exception as e:
        print(f"載入 Cascade 發生例外狀況: {e}")
        return None

# 載入 Haar Cascade
face_cascade = load_cascade_safely('haarcascade_frontalface_alt.xml')

if face_cascade is None:
    print("[錯誤] 無法載入 Haar Cascade 模型，請確認 XML 檔案存在！")
    exit()

# 輸入學生編號/姓名
student_id = input("請輸入學生編號/姓名 (例如 student01): ").strip()
if not student_id:
    student_id = "student01"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "data", student_id)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
max_samples = 30

print(f"開始收集人臉影像，預計收集 {max_samples} 張，按下 'q' 可提前結束...")

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
        count += 1
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        file_path = os.path.join(save_dir, f"{count:02d}.jpg")
        # cv2.imwrite 在中文路徑下可能也會失敗，改用 imencode 寫入
        is_success, buffer = cv2.imencode(".jpg", face_roi)
        if is_success:
            with open(file_path, "wb") as f:
                f.write(buffer)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, f"Saved: {count}/{max_samples}", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    cv2.imshow("Collect Faces - LBPH", frame)

    if cv2.waitKey(100) & 0xFF == ord('q') or count >= max_samples:
        break

cap.release()
cv2.destroyAllWindows()
print(f"成功收集 {count} 張人臉照片至 {save_dir}")