from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

app = FastAPI(title="AEGIS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "AEGIS API is live!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/armor")
async def armor_image(
    file: UploadFile = File(...),
    mode: str = Form("pixel"),
    strength: int = Form(5)
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )

    if mode not in ["pixel", "aml", "max"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid protection mode."
        )

    strength = max(1, min(strength, 15))

    contents = await file.read()

    image_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image."
        )

    original = image.copy()

    # -----------------------------
    # PIXEL PROTECTION
    # -----------------------------
    if mode == "pixel":
        noise = np.random.randint(
            -strength,
            strength + 1,
            image.shape,
            dtype=np.int16
        )

        protected = np.clip(
            image.astype(np.int16) + noise,
            0,
            255
        ).astype(np.uint8)

    # -----------------------------
    # AML-STYLE DEMO
    # -----------------------------
    elif mode == "aml":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            1,
            ksize=3
        )

        edges = np.abs(edges)

        if edges.max() > 0:
            edges = edges / edges.max()

        perturbation = (
            edges[:, :, None] * strength
        )

        noise = np.random.uniform(
            -1,
            1,
            image.shape
        ) * perturbation

        protected = np.clip(
            image.astype(np.float32) + noise,
            0,
            255
        ).astype(np.uint8)

    # -----------------------------
    # AEGIS MAX
    # -----------------------------
    else:
        noise = np.random.randint(
            -strength,
            strength + 1,
            image.shape,
            dtype=np.int16
        )

        protected = np.clip(
            image.astype(np.int16) + noise,
            0,
            255
        ).astype(np.uint8)

        # Small structured perturbation
        blurred = cv2.GaussianBlur(
            original,
            (3, 3),
            0
        )

        detail = (
            original.astype(np.int16)
            - blurred.astype(np.int16)
        )

        protected = np.clip(
            protected.astype(np.int16)
            + (detail // 3),
            0,
            255
        ).astype(np.uint8)

    # Encode as PNG
    success, encoded = cv2.imencode(".png", protected)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Could not encode protected image."
        )

    image_base64 = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return {
        "success": True,
        "mode": mode,
        "strength": strength,
        "media_type": "image/png",
        "image_base64": image_base64
    }
