
import uuid
import logging
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from ocr import extract_text_from_bytes
from language import split_mixed_text
from tts import generate_mixed_speech, AUDIO_DIR
from gan import enhance_audio


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(title="ESP32 Multilingual Reading Assistant")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable or disable GAN enhancement.
ENABLE_GAN = True

AUDIO_DIR = Path(AUDIO_DIR).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ESP32 Multilingual Reading Assistant API",
        "docs": "/docs",
        "gan_enabled": ENABLE_GAN
    }


# --------------------------------------------------
# IMAGE UPLOAD AND PROCESSING
# --------------------------------------------------

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    total_start = time.perf_counter()

    try:
        # 1. Validate uploaded image
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="Please upload an image file."
            )

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty."
            )

        # 2. Extract text using OCR
        ocr_start = time.perf_counter()
        extracted_text = extract_text_from_bytes(image_bytes)
        ocr_time = time.perf_counter() - ocr_start

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text found. Try a clearer image."
            )

        # 3. Split text into language segments
        language_start = time.perf_counter()
        segments = split_mixed_text(extracted_text)
        language_time = time.perf_counter() - language_start

        if not segments:
            raise HTTPException(
                status_code=422,
                detail="No readable language segments were detected."
            )

        # 4. Generate original speech
        audio_name = f"mixed_audio_{uuid.uuid4().hex}.wav"
        audio_path = (AUDIO_DIR / audio_name).resolve()

        tts_start = time.perf_counter()
        generate_mixed_speech(extracted_text, audio_path)
        tts_time = time.perf_counter() - tts_start

        if not audio_path.is_file():
            raise RuntimeError(
                f"TTS did not create the expected audio file: {audio_path}"
            )

        # Keep original audio as the fallback.
        response_audio_name = audio_name
        gan_time = 0.0
        gan_status = "disabled"

        # 5. Optional GAN audio enhancement
        if ENABLE_GAN:
            enhanced_name = f"enhanced_{audio_name}"
            enhanced_path = (AUDIO_DIR / enhanced_name).resolve()

            gan_start = time.perf_counter()

            try:
                result_path = enhance_audio(
                    audio_path,
                    enhanced_path
                )

                result_path = Path(result_path).resolve()

                if not result_path.is_file():
                    raise RuntimeError(
                        f"Enhanced audio file was not created: {result_path}"
                    )

                if result_path.stat().st_size == 0:
                    raise RuntimeError(
                        "Enhanced audio file is empty."
                    )

                # Use enhanced audio only after validation.
                response_audio_name = result_path.name
                gan_status = "success"

            except Exception as error:
                # Log the error and retain the original audio.
                logger.exception(
                    "GAN enhancement failed. Returning original audio."
                )
                gan_status = "failed_using_original_audio"
                logger.error("Exact GAN error: %s", error)

            finally:
                gan_time = time.perf_counter() - gan_start

        # 6. Identify languages present
        languages = list(dict.fromkeys(
            segment["language"] for segment in segments
        ))

        if not languages:
            detected_language = "Unknown"
        elif len(languages) == 1:
            detected_language = languages[0]
        else:
            detected_language = (
                "Mixed (" + ", ".join(languages) + ")"
            )

        # 7. Calculate total processing time
        total_time = time.perf_counter() - total_start

        logger.info("========== PROCESSING TIME ==========")
        logger.info("OCR: %.2f seconds", ocr_time)
        logger.info("Language detection: %.2f seconds", language_time)
        logger.info("TTS: %.2f seconds", tts_time)
        logger.info("GAN: %.2f seconds", gan_time)
        logger.info("GAN status: %s", gan_status)
        logger.info("TOTAL: %.2f seconds", total_time)
        logger.info("=====================================")

        # 8. Return API response
        return {
            "status": "success",
            "filename": file.filename,
            "detected_language": detected_language,
            "extracted_text": extracted_text,
            "language_segments": segments,
            "audio_file": f"/audio/{response_audio_name}",
            "gan_enabled": ENABLE_GAN,
            "gan_status": gan_status,
            "processing_time": {
                "ocr_seconds": round(ocr_time, 2),
                "language_detection_seconds": round(language_time, 2),
                "tts_seconds": round(tts_time, 2),
                "gan_seconds": round(gan_time, 2),
                "total_seconds": round(total_time, 2)
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Image processing failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(error)}"
        )

    finally:
        await file.close()


# --------------------------------------------------
# AUDIO PLAYBACK / DOWNLOAD
# --------------------------------------------------

@app.get("/audio/{filename}")
def get_audio(filename: str):

    # Prevent path traversal.
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio filename."
        )

    audio_path = (AUDIO_DIR / filename).resolve()

    # Ensure the requested file stays inside AUDIO_DIR.
    if audio_path.parent != AUDIO_DIR:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio path."
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found."
        )

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=filename
    )
