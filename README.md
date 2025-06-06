# 🧠 실시간 얼굴 감지 및 블러 처리

> 특정 인물을 클릭하면 그 사람을 제외한 모든 얼굴에 실시간으로 블러 처리되는 시스템

## 📌 주요 기술

- **YOLOv8n-face**: 실시간 얼굴 탐지
- **FaceNet-pytorch**: 얼굴 인식/임베딩
- **OpenCV**: 이미지/비디오 처리, 카메라 제어
- **torchvision.transforms**: 이미지 크기 조정, 텐서 변환, 정규화
- **NumPy**: 배열 연산, 이미지 데이터 처리
- **PyTorch**: 신경망 모델 구성 및 추론

## 🧪 기능

- 처음엔 모두 다 블러 처리 된 상태
- 블러 처리 제외 하고자 하는 인물 누르면 블러 제외
- 마우스로 다른 인물 누르면 그 인물만 블러 제외
- 다른 인물 마우스 클릭시 기존 모자이크 제외 인물 사진 정보 삭제 후 누른 사람의 정보 생김

## 📂 실행 방법

```bash
git clone https://github.com/han-woo-kyeong/face_blur_project.git
cd face_blur_project
pip install -r requirements.txt
# yolov8n-face.pt 파일을 yolov8/ 폴더에 복사
python 얼굴저장_블러효과_최종본.py
