import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageEnhance
from transformers import pipeline

app = FastAPI(
    title="Kisan AI - Indian Village Crop Disease Diagnostic Engine",
    description="High-accuracy vision inference tailored for top Indian regional crops.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multi-crop agricultural vision model
MODEL_ID = "kimcomehome/plantvillage-vit-leaf-disease"
print("⏳ Loading Indian Village Multi-Crop AI Model...")
classifier = pipeline("image-classification", model=MODEL_ID)
print("✅ Multi-Crop Vision Engine loaded and ready!")

# Confidence Thresholds
MIN_CONFIDENCE_THRESHOLD = 45.0   # Reject unrecognizable/non-crop inputs
HIGH_CONFIDENCE_THRESHOLD = 80.0  # Confident field diagnosis

# Recognized Indian Village Crops & Disease Index
VILLAGE_CROPS_REGISTRY = {
    "Paddy": ["Blast (Aggi Tegulu)", "Bacterial Leaf Blight", "Brown Spot", "Sheath Blight", "Healthy"],
    "Cotton": ["Bacterial Blight (Angular Spot)", "Grey Mildew", "Leaf Curl Virus", "Alternaria Spot", "Healthy"],
    "Chilli (Mirchi)": ["Leaf Curl Virus", "Anthracnose / Dieback", "Cercospora Leaf Spot", "Bacterial Spot", "Healthy"],
    "Groundnut": ["Tikka Leaf Spot", "Rust", "Collar Rot", "Healthy"],
    "Sugarcane": ["Red Rot", "Smut", "Grassy Shoot Disease", "Healthy"],
    "Banana": ["Sigatoka Leaf Spot", "Panama Wilt", "Bunchy Top Virus", "Healthy"],
    "Pomegranate": ["Bacterial Blight (Telya)", "Anthracnose", "Healthy"],
    "Papaya": ["Papaya Ring Spot Virus (PRSV)", "Anthracnose", "Healthy"],
    "Wheat": ["Yellow Stripe Rust", "Brown Leaf Rust", "Loose Smut", "Healthy"],
    "Tomato": ["Early Blight", "Late Blight", "Leaf Mold", "Septoria Spot", "Yellow Leaf Curl Virus", "Healthy"],
    "Potato": ["Early Blight", "Late Blight", "Healthy"],
    "Corn (Maize)": ["Common Rust", "Northern Leaf Blight", "Gray Leaf Spot", "Healthy"]
}

def preprocess_leaf(image: Image.Image) -> Image.Image:
    """Normalize orientation and sharpen lesion contrast for mobile farm captures."""
    image = ImageOps.exif_transpose(image)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.15)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "Kisan AI Village Vision Core",
        "focus_crops": list(VILLAGE_CROPS_REGISTRY.keys())
    }

@app.get("/api/v1/crops")
def get_supported_crops():
    return {
        "total_crops": len(VILLAGE_CROPS_REGISTRY),
        "supported_crops": VILLAGE_CROPS_REGISTRY
    }

@app.post("/api/v1/predict/leaf")
async def detect_crop_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_leaf(raw_image)

        # Run multi-class vision inference
        predictions = classifier(processed_image, top_k=5)
        top_match = predictions[0]
        second_match = predictions[1]

        score_1 = round(top_match["score"] * 100, 2)
        score_2 = round(second_match["score"] * 100, 2)

        # Guardrail 1: Non-leaf / low-confidence rejection
        if score_1 < MIN_CONFIDENCE_THRESHOLD:
            return {
                "success": False,
                "status": "UNRELIABLE_SAMPLE",
                "message": "The uploaded photo is unclear or not recognized as a supported crop leaf.",
                "action_required": "Please take a clear close-up picture of the affected leaf in daylight."
            }

        # Format label strings cleanly
        def clean_label(label: str):
            if "___" in label:
                c, d = label.split("___")
                return c.replace("_", " ").replace("Pepper, bell", "Chilli / Pepper").capitalize(), d.replace("_", " ").capitalize()
            return "Plant", label.replace("_", " ").capitalize()

        crop_1, disease_1 = clean_label(top_match["label"])
        crop_2, disease_2 = clean_label(second_match["label"])

        # Guardrail 2: Ambiguity flag
        is_ambiguous = (score_1 - score_2) < 12.0 and score_1 < HIGH_CONFIDENCE_THRESHOLD

        response = {
            "success": True,
            "crop": crop_1,
            "disease_detected": disease_1,
            "confidence_score": f"{score_1}%",
            "certainty_level": "HIGH" if score_1 >= HIGH_CONFIDENCE_THRESHOLD else "MODERATE",
            "is_ambiguous": is_ambiguous,
            "top_possibilities": [
                {
                    "crop": clean_label(p["label"])[0],
                    "disease": clean_label(p["label"])[1],
                    "confidence": f"{round(p['score'] * 100, 2)}%"
                }
                for p in predictions[:3]
            ]
        }

        if is_ambiguous:
            response["field_verification_note"] = (
                f"Close symptoms detected between '{disease_1}' and '{disease_2}'. "
                "Inspect whether spots have yellow rings (bacterial) or powder/mold growth (fungal)."
            )

        return response

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Leaf inference error: {str(error)}")
