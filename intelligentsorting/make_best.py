# make_best.py
import os
from ultralytics import YOLO

def force_create_best_pt():
    print("▶ best.pt 가중치 파일 강제 생성을 시작합니다...")
    
    # 1. YOLOv8의 가장 기본인 Nano 모델을 불러옵니다.
    model = YOLO("yolov8n.pt")
    
    # 2. 딱 1 에폭(epoch)만 돌려서 맛보기 학습을 시킵니다.
    # 데이터셋이 없어도 돌아가도록 임시 가상 환경으로 구동합니다.
    try:
        model.train(
            data="coco8.yaml",  # YOLO 내장 초경량 데모 데이터셋 (다운로드 자동)
            epochs=1,           # 딱 1번만 돌리기
            imgsz=640,
            device="cpu"        # CPU로 안전하게 구동
        )
        print("▶ 가중치 추출 성공!")
    except Exception as e:
        print(f"▶ 에러 발생: {e}")

if __name__ == "__main__":
    force_create_best_pt()