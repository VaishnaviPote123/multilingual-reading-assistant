import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
import wave
from pathlib import Path

from language import (
    split_mixed_text,
    detect_devanagari_language,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
VOICE_DIR = BASE_DIR / "voices"
AUDIO_DIR = BASE_DIR / "audio"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Piper voice models
VOICE_MODELS = {
    "english": "en_US-lessac-medium.onnx",
    "hindi": "hi_IN-pratham-medium.onnx",
    "marathi": "mr_IN-google-medium.onnx",
}


def clean_text(text):
    """Clean text while preserving valid English and Devanagari characters."""
    if not isinstance(text, str):
        text = str(text)

    # Remove invalid Unicode surrogates.
    text = text.encode("utf-8", errors="replace").decode("utf-8")

    # Remove unwanted control characters while preserving whitespace.
    text = "".join(
        char
        for char in text
        if char in "\n\t\r" or ord(char) >= 32
    )

    # Normalize spaces and tabs without removing language characters.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def normalize_language(language, text=""):
    """Convert a language label into a supported Piper voice key."""
    language = (language or "").strip().lower()

    if language in SUPPORTED_LANGUAGES:
        return language

    # Backward compatibility with older language.py labels.
    if language in {
        "hindi_or_marathi",
        "devanagari",
        "unknown",
    }:
        detected = detect_devanagari_language(text)

        if detected in SUPPORTED_LANGUAGES:
            return detected

    raise ValueError(
        f"Unsupported language: {language!r}. "
        f"Expected one of {sorted(SUPPORTED_LANGUAGES)}"
    )


def get_piper_command():
    """Locate Piper on Windows or use its Python module."""
    executable = shutil.which("piper")

    if executable:
        return [executable]

    return [sys.executable, "-m", "piper"]


def validate_wav(output_path):
    """Check that a generated WAV contains valid audio."""
    output_path = Path(output_path)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Piper did not create an audio file: {output_path}"
        )

    try:
        with wave.open(str(output_path), "rb") as wav_file:
            if (
                wav_file.getnchannels() < 1
                or wav_file.getnframes() < 1
                or wav_file.getframerate() < 1
            ):
                raise ValueError("WAV contains no valid audio frames.")

    except (wave.Error, EOFError, ValueError) as exc:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

        raise RuntimeError(
            f"Piper produced an invalid WAV file: {output_path}"
        ) from exc

    return output_path


def generate_segment_speech(text, language, output_path):
    """Generate a single English, Hindi, or Marathi WAV segment."""
    text = clean_text(text)

    if not text:
        raise ValueError("Text is empty after cleaning.")

    language = normalize_language(language, text)

    model_path = VOICE_DIR / VOICE_MODELS[language]
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Voice model not found: {model_path}"
        )

    # Remove stale output before synthesis.
    if output_path.exists():
        output_path.unlink()

    command = get_piper_command() + [
        "--model",
        str(model_path),
        "--output_file",
        str(output_path),
    ]

    # IMPORTANT:
    # PYTHONUTF8=1 fixed the Hindi Unicode error in your direct test.
    # Pass it to Piper every time this function launches a subprocess.
    process_env = os.environ.copy()
    process_env["PYTHONUTF8"] = "1"

    logger.info(
        "Generating %s speech using %s",
        language,
        model_path.name,
    )

    try:
        result = subprocess.run(
            command,
            input=text + "\n",
            capture_output=True,
            check=True,
            cwd=str(BASE_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            env=process_env,
        )

        if result.stderr:
            logger.debug("Piper stderr: %s", result.stderr)

    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr or exc.stdout or str(exc)

        if isinstance(error_message, bytes):
            error_message = error_message.decode(
                "utf-8",
                errors="replace",
            )

        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

        logger.error(
            "Piper failed for %s:\n%s",
            language,
            error_message,
        )

        raise RuntimeError(
            f"Piper failed to generate {language} speech. "
            f"Details: {error_message}"
        ) from exc

    except OSError as exc:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

        logger.exception("Could not start Piper.")

        raise RuntimeError(
            f"Could not start Piper: {exc}"
        ) from exc

    validate_wav(output_path)

    logger.info(
        "Generated %s speech successfully: %s",
        language,
        output_path,
    )

    return output_path


def combine_wav_files(input_paths, output_path):
    """Combine WAV segments with matching audio formats."""
    if not input_paths:
        raise ValueError("No audio segments were generated.")

    input_paths = [Path(path) for path in input_paths]
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        validate_wav(input_path)

    with wave.open(str(input_paths[0]), "rb") as first:
        first_params = first.getparams()

    expected_format = (
        first_params.nchannels,
        first_params.sampwidth,
        first_params.framerate,
        first_params.comptype,
        first_params.compname,
    )

    try:
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(first_params.nchannels)
            output.setsampwidth(first_params.sampwidth)
            output.setframerate(first_params.framerate)
            output.setcomptype(
                first_params.comptype,
                first_params.compname,
            )

            for input_path in input_paths:
                with wave.open(str(input_path), "rb") as source:
                    params = source.getparams()

                    actual_format = (
                        params.nchannels,
                        params.sampwidth,
                        params.framerate,
                        params.comptype,
                        params.compname,
                    )

                    if actual_format != expected_format:
                        raise ValueError(
                            f"WAV formats differ for {input_path}. "
                            "Audio segments must have matching "
                            "channels, sample widths, sample rates, "
                            "and compression formats."
                        )

                    # Append audio frames in the original segment order.
                    output.writeframes(
                        source.readframes(source.getnframes())
                    )

        validate_wav(output_path)

    except Exception:
        # Avoid leaving an incomplete combined audio file.
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

        raise

    logger.info("Combined audio saved: %s", output_path)

    return output_path


def generate_mixed_speech(text, output_path):
    """
    Convert mixed English, Hindi, and Marathi text into one WAV file.

    Maintains compatibility with:
        generate_mixed_speech(extracted_text, audio_path)
    """
    text = clean_text(text)

    if not text:
        raise ValueError("No readable text was provided.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = split_mixed_text(text)

    if not segments:
        raise ValueError(
            "No readable language segments were detected."
        )

    logger.info("Detected %d speech segments.", len(segments))

    segment_paths = []

    try:
        for index, segment in enumerate(segments):
            segment_text = clean_text(segment.get("text", ""))

            if not segment_text:
                continue

            language = normalize_language(
                segment.get("language"),
                segment_text,
            )

            segment_path = (
                AUDIO_DIR
                / f"segment_{uuid.uuid4().hex}_{index}.wav"
            )

            generate_segment_speech(
                segment_text,
                language,
                segment_path,
            )

            segment_paths.append(segment_path)

        if not segment_paths:
            raise RuntimeError(
                "No speech segments were generated."
            )

        combine_wav_files(segment_paths, output_path)

        logger.info(
            "Mixed-language speech generated successfully: %s",
            output_path,
        )

        return str(output_path)

    finally:
        # Delete temporary segments but preserve the final combined WAV.
        for segment_path in segment_paths:
            try:
                segment_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not delete temporary audio file: %s",
                    segment_path,
                )
