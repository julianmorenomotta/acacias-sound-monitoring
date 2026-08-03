import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from inference import detector

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    """Health check endpoint"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@router.post("/predict")
async def predict(audio: UploadFile = File()):
    """Run SED on an uploaded audio file"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model failed to load on startup")

    # Validate file extension
    suffix = Path(audio.filename or "audio.wav").suffix.lower()
    allowed_extensions = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aiff"}
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file formatt '{suffix}'."
            f"Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = detector.predict(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    finally:
        tmp_path_obj = Path(tmp_path) if "tmp_path" in dir() else None
        if tmp_path_obj and tmp_path_obj.exists():
            tmp_path_obj.unlink(missing_ok=True)
