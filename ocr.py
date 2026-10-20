
import logging
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Keep all three languages to support mixed-language pages.
OCR_LANGUAGES = "eng+hin+mar"


def preprocess_image(image):
    """Fast preprocessing without generating multiple image variants."""
    height, width = image.shape[:2]

    # Resize small images to improve character recognition.
    if max(height, width) < 1400:
        image = cv2.resize(
            image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def run_ocr(image, psm=6):
    """Run Tesseract once and reconstruct text lines."""
    config = f"--oem 1 --psm {psm}"

    data = pytesseract.image_to_data(
        image,
        lang=OCR_LANGUAGES,
        config=config,
        output_type=Output.DICT
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
                data["line_num"][i]
            )
            lines.setdefault(key, []).append(word)
            confidences.append(confidence)

    text = "\n".join(
        " ".join(words) for words in lines.values()
    ).strip()

    score = (
        sum(confidences) / len(confidences)
        if confidences else 0.0
    )

    return text, score


def extract_text_from_bytes(image_bytes):
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            "Could not read the image. Upload a valid JPG or PNG."
        )

    processed = preprocess_image(image)

    # First attempt: normal document layout.
    best_text, best_score = run_ocr(processed, psm=6)
    logger.info(
        "OCR pass 1: PSM=6, confidence=%.2f",
        best_score
    )

    # Retry only when the first result appears poor.
    if not best_text or best_score < 45:
        retry_text, retry_score = run_ocr(processed, psm=11)
        logger.info(
            "OCR retry: PSM=11, confidence=%.2f",
            retry_score
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

    logger.info("Final OCR confidence: %.2f", best_score)
    return best_text
