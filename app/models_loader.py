# app/models_loader.py
from pathlib import Path
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper
from ultralytics import YOLO

# new imports
from segment_anything import sam_model_registry, SamPredictor
from transformers import CLIPProcessor, CLIPModel
import easyocr
import os

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Whisper (ASR)
def load_whisper_model(name="small"):
    print("Loading Whisper model:", name)
    return whisper.load_model(name)

# BLIP (captioning)
def load_blip_model(model_name="Salesforce/blip-image-captioning-base"):
    print("Loading BLIP:", model_name)
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

# YOLO
def load_yolo_model(weights="yolov8n.pt"):
    print("Loading YOLO:", weights)
    model = YOLO(weights)
    return model

# SAM (Segment Anything) - requires checkpoint file (see README instructions)
def load_sam(checkpoint_path=None, model_type="default"):
    # model_type picks a key in sam_model_registry; common keys: "vit_h", "vit_l", "vit_b"
    if checkpoint_path is None:
        # look for environment variable SAM_CHECKPOINT or models/sam_vit_h.pth
        env = os.environ.get("SAM_CHECKPOINT")
        if env:
            checkpoint_path = env
        else:
            cp = MODEL_DIR / "sam_vit_h_4b8939.pth"
            checkpoint_path = str(cp) if cp.exists() else None

    if checkpoint_path is None:
        print("Warning: SAM checkpoint not found. SAM will not be available until you download a checkpoint.")
        return None

    print("Loading SAM from", checkpoint_path)
    # choose model type for registry: try vit_h
    sam = sam_model_registry.get("vit_h")(checkpoint=checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

# CLIP for grounding & zero-shot attributes
def load_clip(model_name="openai/clip-vit-base-patch32"):
    print("Loading CLIP:", model_name)
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    return clip_processor, clip_model, device

# EasyOCR reader
def load_ocr(lang_list=["en"]):
    try:
        reader = easyocr.Reader(lang_list, gpu=torch.cuda.is_available())
        return reader
    except Exception as e:
        print("EasyOCR failed to initialize:", e)
        return None

def load_all():
    whisper_m = load_whisper_model("small")
    blip_proc, blip_model, blip_device = load_blip_model()
    yolo_m = load_yolo_model()
    sam_predictor = load_sam()   # may be None until checkpoint downloaded
    clip_proc, clip_model, clip_device = load_clip()
    ocr_reader = load_ocr(["en"])
    return {
        "whisper": whisper_m,
        "blip_proc": blip_proc,
        "blip_model": blip_model,
        "blip_device": blip_device,
        "yolo": yolo_m,
        "sam": sam_predictor,
        "clip_proc": clip_proc,
        "clip_model": clip_model,
        "clip_device": clip_device,
        "ocr": ocr_reader
    }
