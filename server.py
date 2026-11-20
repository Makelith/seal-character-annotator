# /// script
# requires-python = ">=3.10"
# dependencies = ["fastapi", "uvicorn", "ultralytics", "python-multipart", "pillow"]
# ///
from io import BytesIO
from pathlib import Path
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEIGHTS_PATH = Path("model/best.pt")
model = YOLO(str(WEIGHTS_PATH))

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.50),
    iou: float = Form(0.20),
    imgsz: int = Form(1024),
):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    results = model.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    xyxy = results[0].boxes.xyxy.cpu().tolist() if len(results[0].boxes) > 0 else []
    boxes = [[round(c) for c in box] for box in xyxy]
    return {"boxes": boxes}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# uv run server.py