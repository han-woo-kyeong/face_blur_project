# 🧠 실시간 얼굴 감지 및 모자이크 처리

> 특정 인물을 클릭하면 그 사람을 제외한 모든 얼굴에 실시간으로 모자이크 처리되는 시스템

## 📌 주요 기술

- **YOLOv8n-face**: 실시간 얼굴 탐지
- **FaceNet + Cosine Similarity**: 얼굴 유사도 판단
- **OpenCV**: UI 및 후처리
- **face_recognition** (사용시 오류 → 대체됨)

## 🧪 기능

- 처음엔 모두 다 블러 처리 된 상태
- 블러 처리 제외 하고자 하는 인물 누르면 블러 제외
- 마우스로 다른 인물 누르면 그 인물만 블러 제외
- 다른 인물 마우스 클릭시 기존 모자이크 제외 인물 사진 정보 삭제 후 누른 사람의 정보 생김

## 📂 실행 방법

```bash
git clone https://github.com/han-woo-kyeong/face-blur-project.git
cd face-blur-project
pip install -r requirements.txt
python face-mozaic-final.py
