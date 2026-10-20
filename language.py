
import re
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"english", "hindi", "marathi"}

MARATHI_WORDS = {
    "आहे", "आहेत", "आणि", "माझा", "माझी", "माझे",
    "आम्ही", "तुम्ही", "म्हणून", "झाली", "झाले", "झाल्या",
    "महाराष्ट्राची", "महाराष्ट्रात", "पाहिजे", "त्यामुळे",
    "कुठे", "कशी", "कसे", "करण्यासाठी", "यासाठी",
    "त्यांना", "त्याची", "त्याचे", "त्यांच्या", "मुंबईत"
}

HINDI_WORDS = {
    "है", "हैं", "था", "थी", "थे", "और", "यह", "वह",
    "नहीं", "क्योंकि", "लेकिन", "इसलिए", "मेरा", "मेरी",
    "मेरे", "आपका", "आपकी", "करता", "करती", "करते",
    "होगा", "होगी", "होंगे", "हिंदी"
}


def detect_devanagari_language(text):
    """Best-effort classification of Hindi or Marathi text."""
    if not text or not text.strip():
        return "hindi"

    words = set(re.findall(r"[\u0900-\u097F]+", text.lower()))

    marathi_score = len(words & MARATHI_WORDS)
    hindi_score = len(words & HINDI_WORDS)

    if "ळ" in text:
        marathi_score += 2

    if marathi_score > hindi_score and marathi_score > 0:
        return "marathi"

    if hindi_score > marathi_score and hindi_score > 0:
        return "hindi"

    try:
        detected = detect(text)

        if detected == "mr":
            return "marathi"

        if detected == "hi":
            return "hindi"

    except LangDetectException:
        pass

    return "hindi"


def split_mixed_text(text):
    """
    Split mixed English, Hindi and Marathi text into segments.
    Every segment has a supported language name and its text.
    """
    if not text or not text.strip():
        return []

    # Split after sentence-ending punctuation, including । and ॥
    sentences = re.split(r"(?<=[.!?।॥])\s*", text.strip())

    segments = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # Classify each complete sentence independently.
        if re.search(r"[\u0900-\u097F]", sentence):
            language = detect_devanagari_language(sentence)
        elif re.search(r"[A-Za-z]", sentence):
            language = "english"
        else:
            continue

        if language not in SUPPORTED_LANGUAGES:
            language = "hindi"

        segments.append({
            "language": language,
            "text": sentence
        })

    return segments
