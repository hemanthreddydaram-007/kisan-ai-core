import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageEnhance
from transformers import pipeline

app = FastAPI(
    title="Kisan AI - Precision Indian Field Diagnostic Engine",
    description="Precision classification for Indian village staples (Paddy, Cotton, Tomato, Pepper, Potato).",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Crop AI Models...")
hort_classifier = pipeline("image-classification", model="kimcomehome/plantvillage-vit-leaf-disease")

try:
    paddy_classifier = pipeline("image-classification", model="davanstrien/autotrain-rice-disease-classification")
    print("Rice/Paddy Vision Model loaded successfully!")
except Exception:
    paddy_classifier = None
    print("Fallback to multi-crop ensemble.")

print("Vision Pipelines Ready!")

RICE_LABELS_MAP = {
    "bacterial_leaf_blight": "Bacterial Leaf Blight (బాక్టీరియల్ ఆకు ఎండు తెగులు)",
    "brown_spot": "Brown Spot (గోధుమ రంగు మచ్చ తెగులు)",
    "leaf_blast": "Leaf Blast (అగ్గి తెగులు)",
    "sheath_blight": "Sheath Blight (కాండం కుళ్ళు తెగులు)",
    "tungro": "Tungro Virus (తుంగ్రో వైరస్)",
    "healthy": "Healthy Leaf (ఆరోగ్యకరమైన ఆకు)"
}

def preprocess_leaf(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(1.15)

@app.post("/api/v1/predict/leaf")
async def detect_crop_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")
    try:
        image_bytes = await file.read()
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        processed_image = preprocess_leaf(raw_image)

        hort_preds = hort_classifier(processed_image, top_k=3)
        top_hort = hort_preds[0]
        hort_score = round(top_hort["score"] * 100, 2)

        if paddy_classifier:
            rice_preds = paddy_classifier(processed_image, top_k=3)
            top_rice = rice_preds[0]
            rice_score = round(top_rice["score"] * 100, 2)
        else:
            rice_score = 0
            rice_preds = []

        if rice_score > hort_score and rice_score >= 35.0:
            raw_key = top_rice["label"].lower().replace(" ", "_")
            disease_name = RICE_LABELS_MAP.get(raw_key, top_rice["label"].replace("_", " ").title())
            return {
                "success": True,
                "crop": "Paddy / Rice (వరి)",
                "disease_detected": disease_name,
                "confidence_score": f"{rice_score}%",
                "certainty_level": "HIGH" if rice_score >= 75.0 else "MODERATE",
                "is_ambiguous": (rice_score - round(rice_preds[1]["score"] * 100, 2)) < 12.0 if len(rice_preds) > 1 else False,
                "top_possibilities": [
                    {
                        "crop": "Paddy / Rice",
                        "disease": RICE_LABELS_MAP.get(p["label"].lower().replace(" ", "_"), p["label"]),
                        "confidence": f"{round(p["score"] * 100, 2)}%"
                    }
                    for p in rice_preds[:3]
                ]
            }

        if hort_score < 30.0:
            return {
                "success": False,
                "status": "UNRELIABLE_SAMPLE",
                "message": "The leaf image was not recognized with sufficient confidence.",
                "action_required": "Please provide a clearer, closer photo of the infected leaf in natural lighting."
            }

        def clean_label(label: str):
            if "___" in label:
                c, d = label.split("___")
                return c.replace("_", " ").replace("Pepper, bell", "Chilli / Capsicum").capitalize(), d.replace("_", " ").capitalize()
            return "Plant", label.replace("_", " ").capitalize()

        crop_1, dis_1 = clean_label(top_hort["label"])
        return {
            "success": True,
            "crop": crop_1,
            "disease_detected": dis_1,
            "confidence_score": f"{hort_score}%",
            "certainty_level": "HIGH" if hort_score >= 75.0 else "MODERATE",
            "is_ambiguous": False,
            "top_possibilities": [
                {
                    "crop": clean_label(p["label"])[0],
                    "disease": clean_label(p["label"])[1],
                    "confidence": f"{round(p["score"] * 100, 2)}%"
                }
                for p in hort_preds[:3]
            ]
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(error)}")
