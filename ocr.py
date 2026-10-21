
import logging
import os
import shutil

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)


# --------------------------------------------------
# TESSERACT CONFIGURATION
# --------------------------------------------------

if os.name == "nt":
    # Windows: local development in VS Code
    windows_tesseract = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    if os.path.isfile(windows_tesseract):
        pytesseract.pytesseract.tesseract_cmd = windows_tesseract
    else:
        # Fall back to Tesseract available on Windows PATH
        detected_tesseract = shutil.which("tesseract")

        if detected_tesseract:
            pytesseract.pytesseract.tesseract_cmd = (
                detected_tesseract
            )
        else:
            raise RuntimeError(
                "Tesseract OCR was not found. Install Tesseract "
                "or add it to your Windows PATH."
            )

else:
    # Linux: Docker and Render
    detected_tesseract = shutil.which("tesseract")

    if detected_tesseract:
        pytesseract.pytesseract.tesseract_cmd = (
            detected_tesseract
        )
    else:
        pytesseract.pytesseract.tesseract_cmd = (
            "/usr/bin/tesseract"
        )

        if not os.path.isfile("/usr/bin/tesseract"):
            raise RuntimeError(
                "Tesseract OCR is not installed in this environment."
            )


# --------------------------------------------------
# OCR LANGUAGE CONFIGURATION
# --------------------------------------------------

OCR_LANGUAGES = "eng+hin+mar"


def verify_ocr_setup():
    """Verify Tesseract and the required language data."""
    try:
        version = pytesseract.get_tesseract_version()
        installed_languages = set(
            pytesseract.get_languages(config="")
        )

        required_languages = {"eng", "hin", "mar"}
        missing_languages = (
            required_languages - installed_languages
        )

        if missing_languages:
            raise RuntimeError(
                "Missing Tesseract language data: "
                + ", ".join(sorted(missing_languages))
            )

        logger.info("Tesseract version: %s", version)
        logger.info(
            "Required OCR languages are installed: %s",
            OCR_LANGUAGES,
        )

    except Exception:
        logger.exception("Tesseract OCR setup verification failed")
        raise


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(image):
    """Convert the image to grayscale and enlarge small images."""
    height, width = image.shape[:2]

    # Enlarge small images to improve character recognition.
    if max(height, width) < 1400:
        image = cv2.resize(
            image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray


# --------------------------------------------------
# OCR PROCESSING
# --------------------------------------------------

def run_ocr(image, psm=6):
    """Extract text and calculate average OCR confidence."""
    config = f"--oem 1 --psm {psm}"

    data = pytesseract.image_to_data(
        image,
        lang=OCR_LANGUAGES,
        config=config,
        output_type=Output.DICT,
    )

    lines = {}
    confidences = []

    for i, raw_word in enumerate(data["text"]):
        word = raw_word.strip()

        try:
            confidence = float(data["conf"][i])
        except (TypeError, ValueError):
            continue

        if word and confidence >= 0:
            key = (
                data["block_num"][i],
                data["par_num"][i],
                data["line_num"][i],
            )

            lines.setdefault(key, []).append(word)
            confidences.append(confidence)

    text = "\n".join(
        " ".join(words)
        for words in lines.values()
    ).strip()

    score = (
        sum(confidences) / len(confidences)
        if confidences
        else 0.0
    )

    return text, score


# --------------------------------------------------
# PUBLIC IMAGE-TO-TEXT FUNCTION
# --------------------------------------------------

def extract_text_from_bytes(image_bytes):
    """Extract English, Hindi, and Marathi text from image bytes."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            "Could not read the image. Upload a valid JPG or PNG."
        )

    processed = preprocess_image(image)

    # First attempt: standard document layout.
    best_text, best_score = run_ocr(processed, psm=6)

    logger.info(
        "OCR pass 1: PSM=6, confidence=%.2f",
        best_score,
    )

    # Retry if no text is detected or confidence is low.
    if not best_text or best_score < 45:
        retry_text, retry_score = run_ocr(processed, psm=11)

        logger.info(
            "OCR retry: PSM=11, confidence=%.2f",
            retry_score,
        )

        if retry_text and (
            not best_text or retry_score > best_score
        ):
            best_text = retry_text
            best_score = retry_score

    if not best_text:
        raise RuntimeError(
            "No text detected. Try a clearer image with better lighting."
        )

    logger.info(
        "Final OCR confidence: %.2f",
        best_score,
    )

    return best_text
