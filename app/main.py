# app/main.py
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import cv2
import numpy as np

# --- Local imports ---
from .models_loader import load_all
from .utils import save_image, save_audio, save_metadata, load_metadata, IMAGES_DIR

# ----------------------------
# APP INIT
# ----------------------------
app = FastAPI(title="Multimodal Assistant API")

# ----------------------------
# MODEL LOADING
# ----------------------------
print("Loading models (this may take a while)...")
models = load_all()
print("✅ Models loaded successfully!")

# ----------------------------
# STATIC UI
# ----------------------------
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    print(f"⚠️ Static directory not found: {STATIC_DIR}")
# ----------------------------
# HELPERS
# ----------------------------
def generate_caption(image_path, processor, model, device, max_length=50):
    """Generate caption using BLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_length)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def detect_people(image_path, yolo_model, conf=0.25):
    """Detect people and objects using YOLO."""
    results = yolo_model(image_path, imgsz=640, conf=conf)[0]
    boxes, person_count = [], 0
    for box, cls, conf in zip(
        results.boxes.xyxy.tolist(),
        results.boxes.cls.tolist(),
        results.boxes.conf.tolist()
    ):
        class_id = int(cls)
        label = results.names[class_id] if hasattr(results, "names") else str(class_id)
        x1, y1, x2, y2 = [float(x) for x in box]
        boxes.append({"label": label, "conf": float(conf), "bbox": [x1, y1, x2, y2]})
        if label.lower() in ["person", "people"]:
            person_count += 1
    return person_count, boxes


def transcribe_audio(audio_path, whisper_model):
    """Transcribe audio question using Whisper."""
    result = whisper_model.transcribe(audio_path)
    return result.get("text", ""), result


def draw_boxes(image_path, boxes, out_dir="data/annotated"):
    """Draw bounding boxes and labels on the image."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["bbox"])
        label = b.get("label", "")
        conf = b.get("conf", 0)
        color = (0, 255, 0) if label.lower() == "person" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    out_path = Path(out_dir) / (Path(image_path).stem + "_annotated.jpg")
    cv2.imwrite(str(out_path), img)
    return str(out_path)

# ----------------------------
# ROUTES (PREFIXED WITH /api)
# ----------------------------
@app.get("/api/ping")
async def ping():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image, caption it, detect objects, and annotate."""
    saved_path, filename = save_image(file)
    caption = generate_caption(
        saved_path, models["blip_proc"], models["blip_model"], models["blip_device"]
    )
    person_count, boxes = detect_people(saved_path, models["yolo"])

    # Create annotated version
    annotated_path = draw_boxes(saved_path, boxes)

    metadata = {
        "filename": filename,
        "caption": caption,
        "person_count": person_count,
        "detection_boxes": boxes,
        "annotated_path": annotated_path,
    }
    save_metadata(filename, metadata)
    return JSONResponse({"status": "ok", "image_id": filename, "metadata": metadata})


@app.post("/api/ask")
async def ask(image_id: str = Form(...), audio: UploadFile = File(None), text: str = Form(None)):
    """Handle questions (text or voice) about the uploaded image."""
    meta = load_metadata(image_id)
    if meta is None:
        return JSONResponse({"status": "error", "message": "image not found"}, status_code=404)

    # Get question text
    if audio is not None:
        audio_path, _ = save_audio(audio)
        question_text, raw_whisper = transcribe_audio(audio_path, models["whisper"])
    elif text:
        question_text, raw_whisper = text, {}
    else:
        return JSONResponse({"status": "error", "message": "no question provided"}, status_code=400)

    q_lower = question_text.lower()

    if any(k in q_lower for k in ["how many", "count", "number of people", "people are there"]):
        answer = f"I detect {meta['person_count']} people in the image."
        evidence = {"person_count": meta['person_count']}
    elif any(k in q_lower for k in ["what is in the image", "describe", "what is happening"]):
        answer = meta.get("caption", "I can't describe this image.")
        evidence = {"caption": answer}
    else:
        answer = f"{meta['caption']} I also detect {meta['person_count']} people."
        evidence = {"caption": meta['caption'], "person_count": meta['person_count']}

    response = {
        "status": "ok",
        "question": question_text,
        "answer": answer,
        "evidence": evidence,
        "transcription": question_text,
    }
    return JSONResponse(response)


@app.get("/api/image/{image_id}")
async def get_image(image_id: str):
    """Return the original uploaded image."""
    path = IMAGES_DIR / image_id
    if not path.exists():
        return JSONResponse({"status": "error", "message": "not found"}, status_code=404)
    return FileResponse(str(path))

# ----------------------------
# STATIC ANNOTATED IMAGE SERVING
# ----------------------------
ANNOTATED_DIR = Path(__file__).resolve().parent.parent / "data" / "annotated"
app.mount("/annotated", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated")

# ----------------------------
# MAIN ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
