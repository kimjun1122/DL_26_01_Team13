# train.py
from ultralytics import YOLO

if __name__ == "__main__":
    # 기본 가벼운 모델 로드
    model = YOLO("yolov8n.pt")

    # 컴퓨터 사양과 파일 수에 맞게 30번 학습 진행
    model.train(data="data.yaml", epochs=1, imgsz=640, device='cpu')