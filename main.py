
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

# Configure logging before importing project modules.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)

logger = logging.getLogger(__name__)

from ocr import extract_text_from_bytes
from language import split_mixed_text
from tts import generate_mixed_speech, AUDIO_DIR
from gan import enhance_audio


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(title="ESP32 Multilingual Reading Assistant")

ENABLE_GAN = True

AUDIO_DIR = Path(AUDIO_DIR).resolve()
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Application initialized")
logger.info("Audio directory: %s", AUDIO_DIR)
logger.info("GAN enhancement enabled: %s", ENABLE_GAN)


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "ESP32 Multilingual Reading Assistant API",
        "docs": "/docs",
        "gan_enabled": ENABLE_GAN,
    }


# --------------------------------------------------
# IMAGE UPLOAD AND PROCESSING
# --------------------------------------------------

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    request_id = uuid.uuid4().hex[:12]
    total_start = time.perf_counter()

    logger.info(
        "[%s] Upload request received | filename=%s | content_type=%s",
        request_id,
        file.filename,
        file.content_type,
    )

    try:
        # 1. Validate upload
        if not file.content_type or not file.content_type.startswith("image/"):
            logger.warning("[%s] Rejected: uploaded file is not an image", request_id)
            raise HTTPException(
                status_code=400,
                detail="Please upload an image file.",
            )

        image_bytes = await file.read()

        if not image_bytes:
            logger.warning("[%s] Rejected: uploaded image is empty", request_id)
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        logger.info(
            "[%s] Image received | size=%.2f KB",
            request_id,
            len(image_bytes) / 1024,
        )

        # 2. OCR
        logger.info("[%s] OCR started", request_id)
        ocr_start = time.perf_counter()

        extracted_text = extract_text_from_bytes(image_bytes)

        ocr_time = time.perf_counter() - ocr_start

        logger.info(
            "[%s] OCR finished | elapsed=%.2fs | characters=%d",
            request_id,
            ocr_time,
            len(extracted_text or ""),
        )

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text found. Try a clearer image.",
            )

        # 3. Language detection and segmentation
        logger.info("[%s] Language segmentation started", request_id)
        language_start = time.perf_counter()

        segments = split_mixed_text(extracted_text)

        language_time = time.perf_counter() - language_start

        logger.info(
            "[%s] Language segmentation finished | elapsed=%.2fs | segments=%d",
            request_id,
            language_time,
            len(segments) if segments else 0,
        )

        if not segments:
            raise HTTPException(
                status_code=422,
                detail="No readable language segments were detected.",
            )

        # 4. Text-to-speech
        audio_name = f"mixed_audio_{uuid.uuid4().hex}.wav"
        audio_path = (AUDIO_DIR / audio_name).resolve()

        if audio_path.parent != AUDIO_DIR:
            raise RuntimeError("Generated audio path is outside the audio directory.")

        logger.info("[%s] TTS started | output=%s", request_id, audio_name)
        tts_start = time.perf_counter()

        generate_mixed_speech(extracted_text, audio_path)

        tts_time = time.perf_counter() - tts_start

        logger.info(
            "[%s] TTS finished | elapsed=%.2fs",
            request_id,
            tts_time,
        )

        if not audio_path.is_file():
            raise RuntimeError(
                f"TTS did not create the expected audio file: {audio_path}"
            )

        if audio_path.stat().st_size == 0:
            raise RuntimeError("TTS created an empty audio file.")

        logger.info(
            "[%s] Original audio verified | size=%.2f KB",
            request_id,
            audio_path.stat().st_size / 1024,
        )

        # Keep original audio as the fallback.
        response_audio_name = audio_name
        gan_time = 0.0
        gan_status = "disabled"

        # 5. Optional GAN enhancement
        if ENABLE_GAN:
            enhanced_name = f"enhanced_{audio_name}"
            enhanced_path = (AUDIO_DIR / enhanced_name).resolve()

            logger.info("[%s] GAN enhancement started", request_id)
            gan_start = time.perf_counter()

            try:
                result_path = enhance_audio(audio_path, enhanced_path)

                if result_path is None:
                    raise RuntimeError(
                        "GAN enhancement did not return an audio path."
                    )

                result_path = Path(result_path).resolve()

                if result_path.parent != AUDIO_DIR:
                    raise RuntimeError(
                        "Enhanced audio path is outside the audio directory."
                    )

                if not result_path.is_file():
                    raise RuntimeError(
                        f"Enhanced audio file was not created: {result_path}"
                    )

                if result_path.stat().st_size == 0:
                    raise RuntimeError("Enhanced audio file is empty.")

                response_audio_name = result_path.name
                gan_status = "success"

                logger.info(
                    "[%s] GAN enhancement succeeded | output=%s",
                    request_id,
                    response_audio_name,
                )

            except Exception:
                logger.exception(
                    "[%s] GAN enhancement failed; using original audio",
                    request_id,
                )
                gan_status = "failed_using_original_audio"

            finally:
                gan_time = time.perf_counter() - gan_start
                logger.info(
                    "[%s] GAN stage finished | elapsed=%.2fs | status=%s",
                    request_id,
                    gan_time,
                    gan_status,
                )

        # 6. Determine detected languages
        languages = list(
            dict.fromkeys(
                segment["language"]
                for segment in segments
                if segment.get("language")
            )
        )

        if not languages:
            detected_language = "Unknown"
        elif len(languages) == 1:
            detected_language = languages[0]
        else:
            detected_language = "Mixed (" + ", ".join(languages) + ")"

        # 7. Confirm final audio is available
        final_audio_path = (AUDIO_DIR / response_audio_name).resolve()

        if final_audio_path.parent != AUDIO_DIR:
            raise RuntimeError("Final audio path is invalid.")

        if not final_audio_path.is_file():
            raise RuntimeError("Final audio file is missing.")

        # 8. Processing metrics
        total_time = time.perf_counter() - total_start

        logger.info(
            "[%s] PROCESSING COMPLETE | OCR=%.2fs | LANGUAGE=%.2fs "
            "| TTS=%.2fs | GAN=%.2fs | TOTAL=%.2fs",
            request_id,
            ocr_time,
            language_time,
            tts_time,
            gan_time,
            total_time,
        )

        # 9. API response
        return {
            "status": "success",
            "request_id": request_id,
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
                "total_seconds": round(total_time, 2),
            },
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception(
            "[%s] Image processing failed | type=%s",
            request_id,
            type(error).__name__,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {type(error).__name__}: {str(error)}",
        ) from error

    finally:
        await file.close()
        logger.info(
            "[%s] Upload request finished | elapsed=%.2fs",
            request_id,
            time.perf_counter() - total_start,
        )


# --------------------------------------------------
# AUDIO PLAYBACK / DOWNLOAD
# --------------------------------------------------

@app.get("/audio/{filename}")
def get_audio(filename: str):

    # Prevent path traversal.
    if Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio filename.",
        )

    audio_path = (AUDIO_DIR / filename).resolve()

    # Ensure the resolved path remains directly inside AUDIO_DIR.
    if audio_path.parent != AUDIO_DIR:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio path.",
        )

    if not audio_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Audio file not found.",
        )

    logger.info("Audio requested | filename=%s", filename)

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=filename,
    )
