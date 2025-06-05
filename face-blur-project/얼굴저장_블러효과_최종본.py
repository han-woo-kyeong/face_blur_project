import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# 모델 로드(모델 경로는 사용되는 환경에 따라 경로 달라짐.)
model = YOLO("C:/Users/woo/face-blur-project/Yolov8/yolov8n-face.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 전처리
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 디렉토리 설정
save_dir = "collected_faces"
os.makedirs(save_dir, exist_ok=True)
save_count = 0

# 얼굴 관련 변수
faces_detected = []
selected_face_embedding = None
frame = None

def get_face_embedding(face_img):
    try:
        img = cv2.resize(face_img, (160, 160))
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = facenet(img_tensor)
        return embedding
    except:
        return None

def click_event(event, x, y, flags, param):
    global selected_face_embedding, frame, save_count
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (x1, y1, x2, y2) in enumerate(faces_detected):
            if x1 < x < x2 and y1 < y < y2:
                face_img = frame[y1:y2, x1:x2]

                # 저장 디렉토리 초기화
                for file in os.listdir(save_dir):
                    os.remove(os.path.join(save_dir, file))
                save_count = 0

                filename = os.path.join(save_dir, f"face_{save_count:04d}.jpg")
                cv2.imwrite(filename, face_img)
                save_count += 1
                print(f"선택된 얼굴 저장됨: {filename}")

                selected_face_embedding = get_face_embedding(face_img)
                if selected_face_embedding is not None:
                    print("선택된 얼굴 임베딩 저장 완료")
                else:
                    print("임베딩 추출 실패")

# 창 설정
cv2.namedWindow("Face Recognition")
cv2.setMouseCallback("Face Recognition", click_event)

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 오류")
        break

    results = model(frame)
    faces_detected.clear()

    # 얼굴 박스 저장
    for result in results[0].boxes.xywh:
        x_center, y_center, w, h = map(int, result[:4])
        x1 = max(int(x_center - w / 2), 0)
        y1 = max(int(y_center - h / 2), 0)
        x2 = min(int(x_center + w / 2), frame.shape[1])
        y2 = min(int(y_center + h / 2), frame.shape[0])
        faces_detected.append((x1, y1, x2, y2))

    selected_index = -1
    min_dist = float('inf')

    # 선택된 얼굴 임베딩과 비교
    if selected_face_embedding is not None:
        for i, (x1, y1, x2, y2) in enumerate(faces_detected):
            face = frame[y1:y2, x1:x2]
            embedding = get_face_embedding(face)
            if embedding is not None:
                dist = torch.norm(selected_face_embedding - embedding).item()
                if dist < min_dist:
                    min_dist = dist
                    selected_index = i

        if min_dist > 1.0:  # 너무 다르면 무시
            selected_index = -1

    # 얼굴 표시
    for i, (x1, y1, x2, y2) in enumerate(faces_detected):
        if i == selected_index:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            face = frame[y1:y2, x1:x2]
            if face.size != 0:
                # 블러 처리
                ksize = (31, 31)  # 커널 크기, 홀수로 설정
                blurred_face = cv2.GaussianBlur(face, ksize, 0)
                frame[y1:y2, x1:x2] = blurred_face


    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
