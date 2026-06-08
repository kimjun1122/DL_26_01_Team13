import cv2
import numpy as np
import base64
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI()

# 1. 학습된 모델 로드
try:
    model = YOLO("best.pt") # 직접 학습시킨 파일
except:
    model = YOLO("yolov8n.pt") # 없을 경우 기본 모델

# 2. 분리배출 DB (AI Hub 클래스명과 매칭 필수)
WASTE_DB = {
    "bottle": {"name": "투명 페트병", "guide": "라벨 제거 후 세척하여 배출!", "points": 50},
    "can": {"name": "캔류", "guide": "내용물 비우고 압착하여 배출!", "points": 40},
    "plastic": {"name": "플라스틱", "guide": "오염물 제거 후 재질별로 배출!", "points": 30},
    "paper": {"name": "종이류", "guide": "테이프 제거 후 펼쳐서 배출!", "points": 20}
}

class ImageInput(BaseModel):
    image: str

@app.post("/detect")
async def detect_api(data: ImageInput):
    try:
        encoded_data = data.image.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        results = model(img, conf=0.4)[0]
        
        detections = []
        for box in results.boxes:
            label = model.names[int(box.cls[0])].lower()
            info = WASTE_DB.get(label, {"name": "미분류", "guide": "일반 쓰레기로 배출하세요.", "points": 0})
            detections.append({"name": info["name"], "guide": info["guide"], "points": info["points"]})
        return {"success": True, "results": detections}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <title>지능형 분리배출 시스템</title>
        <style>
            body { background-color: #f0fdf4; font-family: sans-serif; }
            .mint-card { background: white; border-radius: 2.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.05); }
        </style>
    </head>
    <body class="flex flex-col items-center p-6 min-h-screen">
        <div class="max-w-md w-full mint-card p-8 text-center mt-10">
            <h1 class="text-2xl font-bold text-gray-800 mb-2">분리배출 가이드</h1>
            <p class="text-emerald-500 font-bold mb-8">AI Intelligent Guide</p>
            
            <div class="relative bg-gray-100 rounded-[2rem] aspect-[4/5] mb-6 overflow-hidden border-2 border-emerald-50">
                <img id="preview" class="hidden w-full h-full object-cover">
                <div id="loading" class="absolute inset-0 bg-black/40 flex items-center justify-center text-white hidden">
                    <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-white"></div>
                </div>
            </div>

            <input type="file" id="cam" accept="image/*" capture="environment" class="hidden" onchange="upload(event)">
            <button onclick="document.getElementById('cam').click()" class="w-full bg-[#2ecc71] text-white font-bold py-5 rounded-2xl text-xl shadow-lg active:scale-95 transition">
                📸 사진 찍기
            </button>

            <div id="res" class="mt-8 text-left hidden bg-emerald-50 p-6 rounded-[2rem] border border-emerald-100">
                <h2 id="resName" class="text-xl font-bold text-gray-800"></h2>
                <p id="resGuide" class="text-gray-600 text-sm mt-3"></p>
                <p id="resPts" class="text-emerald-600 font-bold mt-4 pt-4 border-t border-emerald-200 text-right"></p>
            </div>
        </div>
    </body>
    <script>
        async function upload(e) {
            const file = e.target.files[0]; if(!file) return;
            const preview = document.getElementById('preview');
            const loading = document.getElementById('loading');
            const resBox = document.getElementById('res');
            
            preview.src = URL.createObjectURL(file);
            preview.classList.remove('hidden');
            loading.classList.remove('hidden');
            resBox.classList.add('hidden');

            const reader = new FileReader();
            reader.onload = async (ev) => {
                const response = await fetch('/detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image: ev.target.result})
                });
                const data = await response.json();
                loading.classList.add('hidden');
                if(data.success && data.results.length > 0) {
                    const item = data.results[0];
                    resBox.classList.remove('hidden');
                    document.getElementById('resName').innerText = item.name;
                    document.getElementById('resGuide').innerText = item.guide;
                    document.getElementById('resPts').innerText = "+" + item.points + "P 적립";
                }
            };
            reader.readAsDataURL(file);
        }
    </script>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)