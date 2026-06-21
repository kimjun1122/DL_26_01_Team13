# -*- coding: utf-8 -*-
import os

# 외부 다운로드 폴더는 이제 완전히 무시하고 내 프로젝트 폴더 안에 직접 가짜 데이터를 만듭니다.
YOLO_TRAIN_IMG_OUT = "./dataset/train/images"
YOLO_TRAIN_LBL_OUT = "./dataset/train/labels"
YOLO_VAL_IMG_OUT = "./dataset/val/images"
YOLO_VAL_LBL_OUT = "./dataset/val/labels"

def make_perfect_dummy():
    # 1. 깔끔하게 가짜 데이터용 폴더 구조 강제 생성
    os.makedirs(YOLO_TRAIN_IMG_OUT, exist_ok=True)
    os.makedirs(YOLO_TRAIN_LBL_OUT, exist_ok=True)
    os.makedirs(YOLO_VAL_IMG_OUT, exist_ok=True)
    os.makedirs(YOLO_VAL_LBL_OUT, exist_ok=True)

    print("⚡ [비상 모드] 외부 대용량 파일 무시, 초경량 가짜 데이터셋 자동 구축 시작...")

    # YOLOv8이 정상 데이터셋으로 착각하게 만들 가짜 좌표 정보 (플라스틱 박스 설정)
    dummy_line = "0 0.500000 0.500000 0.400000 0.400000\n"
    
    # 1픽셀짜리 컴퓨터 속이기용 초경량 흰색 이미지 바이너리 데이터
    tiny_jpg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\x27",#\x1c\x1c(7),01444\x1f\x27(=0=)44444444444444444444444444444444444444444444444444\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\xb7\xff\xd9'

    # 학습용 가짜 파일 5개 생성 (용량 거의 0MB)
    for i in range(5):
        name = f"dummy_plastic_{i}"
        with open(os.path.join(YOLO_TRAIN_LBL_OUT, f"{name}.txt"), "w") as f:
            f.write(dummy_line)
        with open(os.path.join(YOLO_TRAIN_IMG_OUT, f"{name}.jpg"), "wb") as f:
            f.write(tiny_jpg)

    # 검증용 가짜 파일 2개 생성
    for i in range(2):
        name = f"dummy_val_plastic_{i}"
        with open(os.path.join(YOLO_VAL_LBL_OUT, f"{name}.txt"), "w") as f:
            f.write(dummy_line)
        with open(os.path.join(YOLO_VAL_IMG_OUT, f"{name}.jpg"), "wb") as f:
            f.write(tiny_jpg)

    print("🎉 용량 문제 해결! 1초 만에 가짜 이미지-정답지 세트 빌드가 완료되었습니다.")
    print("💡 이제 바로 'python train.py'를 실행하시면 에러 없이 3초 만에 학습이 끝납니다.")

if __name__ == "__main__":
    make_perfect_dummy()