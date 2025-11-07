# app/utils.py
import os
import json
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO
import torch
from sklearn.cluster import KMeans
import threading
import pyttsx3
import time
# ---------- Core data directories ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
META_DIR = DATA_DIR / "metadata"
ANNOTATED_DIR = DATA_DIR / "annotated"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Basic save/load utilities ----------

def save_image(file, filename=None):
    """Save uploaded image file and return path + filename"""
    if filename is None:
        filename = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + "_" + file.filename
    out_path = IMAGES_DIR / filename
    with open(out_path, "wb") as f:
        f.write(file.file.read())
    return str(out_path), filename

def save_audio(file, filename=None):
    """Save uploaded audio file and return path + filename"""
    if filename is None:
        filename = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + "_" + file.filename
    out_path = DATA_DIR / filename
    with open(out_path, "wb") as f:
        f.write(file.file.read())
    return str(out_path), filename

def save_metadata(image_filename, data: dict):
    """Store JSON metadata for a processed image"""
    meta_path = META_DIR / (image_filename + ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(meta_path)

def load_metadata(image_filename):
    """Load saved JSON metadata if available"""
    meta_path = META_DIR / (image_filename + ".json")
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- Visualization helpers ----------

def draw_boxes_and_masks(image_path, boxes, masks=None, out_name=None):
    """Draw bounding boxes and (optional) masks, save annotated image."""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # draw masks
    if masks:
        for m in masks:
            mask = m.get("mask")
            color = m.get("color", (255, 0, 0, 80))
            if mask is None:
                continue
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
            colored = Image.new("RGBA", img.size, color)
            overlay.paste(colored, (0, 0), mask_img)

    # draw boxes and labels
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        label = f"{b.get('label','obj')}:{b.get('conf',0):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0, 255), width=2)
        draw.text((x1 + 3, y1 + 3), label, fill=(255, 255, 255, 255), font=font)

    combined = Image.alpha_composite(img, overlay).convert("RGB")
    if out_name is None:
        out_name = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + "_annot.jpg"
    out_path = ANNOTATED_DIR / out_name
    combined.save(out_path, quality=90)
    return str(out_path)

# ---------- Color and attribute analysis ----------

def dominant_color_in_bbox(image_path, bbox):
    """Extract dominant color within a bounding box region."""
    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = [int(x) for x in bbox]
    cropped = image.crop((x1, y1, x2, y2)).resize((64, 64))
    arr = np.asarray(cropped).reshape(-1, 3)
    km = KMeans(n_clusters=3, random_state=0).fit(arr)
    counts = np.bincount(km.labels_)
    dominant = km.cluster_centers_[np.argmax(counts)].astype(int)

    def rgb_to_name(rgb):
        r, g, b = rgb
        if r > 200 and g < 80 and b < 80: return "red"
        if r > 200 and g > 200 and b < 100: return "yellow"
        if r < 80 and g > 150 and b < 80: return "green"
        if r < 100 and g < 100 and b > 150: return "blue"
        if r > 200 and g > 200 and b > 200: return "white"
        if r < 60 and g < 60 and b < 60: return "black"
        if r > 150 and g > 80 and b < 80: return "brown"
        return "mixed"

    return {"rgb": dominant.tolist(), "color_name": rgb_to_name(dominant.tolist())}

# ---------- CLIP zero-shot classification ----------

def clip_zero_shot_classify(clip_proc, clip_model, device, image_pil, candidate_labels):
    """Zero-shot classification using CLIP over candidate labels."""
    inputs = clip_proc(text=candidate_labels, images=image_pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits_per_image = clip_model(**inputs).logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    top_idx = int(probs.argmax())
    return {
        "label": candidate_labels[top_idx],
        "score": float(probs[top_idx]),
        "all": list(zip(candidate_labels, probs.tolist()))
    }

# ---------- SAM mask generation ----------
# ---------- TTS helpers (pyttsx3) ----------
TTS_DIR = DATA_DIR / "tts"
TTS_DIR.mkdir(parents=True, exist_ok=True)

# pyttsx3 engine is not thread-safe across many calls; create helper to synthesize to file
def synthesize_tts_pyttsx3(text, out_filename=None, voice=None, rate=None):
    """
    Synthesize `text` to a WAV file using pyttsx3.
    Returns the full path to the generated WAV file.
    """
    if out_filename is None:
        out_filename = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + "_tts.wav"
    out_path = TTS_DIR / out_filename

    # Create engine per call to avoid threading issues
    engine = pyttsx3.init()
    if voice:
        try:
            engine.setProperty('voice', voice)
        except Exception:
            pass
    if rate:
        try:
            engine.setProperty('rate', int(rate))
        except Exception:
            pass

    # pyttsx3 can save to file
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    # small sleep to ensure file is flushed
    time.sleep(0.2)
    return str(out_path)
def sam_predict_masks(sam_predictor, image_path, boxes=None):
    """Generate segmentation masks for boxes using SAM predictor."""
    if sam_predictor is None:
        return None
    image = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    sam_predictor.set_image(image)
    results = []
    if boxes:
        xyxy = np.array(boxes)
        xywh = np.stack([xyxy[:, 0], xyxy[:, 1],
                         xyxy[:, 2] - xyxy[:, 0],
                         xyxy[:, 3] - xyxy[:, 1]], axis=1)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(torch.tensor(xywh), image.shape[:2])
        masks, scores, logits = sam_predictor.predict_torch(boxes=transformed_boxes, multimask_output=False)
        for m in masks:
            m_np = m.cpu().numpy().astype(bool)
            results.append(m_np)
    else:
        return None
    return results
