
import logging
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.enhancement import SpectralMaskEnhancement


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"

MODEL_DIR = (
    BASE_DIR
    / "pretrained_models"
    / "metricgan-plus-voicebank"
)

MODEL_SOURCE = "speechbrain/metricgan-plus-voicebank"
TARGET_SAMPLE_RATE = 16000

logger = logging.getLogger(__name__)

# Load the model only once per Python process.
_enhancer = None


# ==================================================
# LOAD METRICGAN+ MODEL
# ==================================================

def load_gan_model():
    """
    Load the pretrained SpeechBrain MetricGAN+ model.
    The model is cached after the first successful load.
    """
    global _enhancer

    if _enhancer is not None:
        return _enhancer

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading MetricGAN+ model...")

    _enhancer = SpectralMaskEnhancement.from_hparams(
        source=MODEL_SOURCE,
        savedir=str(MODEL_DIR),
        run_opts={"device": "cpu"},
    )

    logger.info("MetricGAN+ model loaded successfully.")

    return _enhancer


# ==================================================
# AUDIO PREPROCESSING
# ==================================================

def load_audio_file(audio_path):
    """
    Read WAV audio using SoundFile rather than torchaudio.
    Convert to mono and resample to 16 kHz when required.
    """
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )

    if audio.size == 0:
        raise ValueError("The input WAV file contains no audio.")

    # Convert stereo or multichannel audio to mono.
    audio = np.mean(audio, axis=1).astype(np.float32)

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")

    # MetricGAN+ VoiceBank is intended for 16 kHz audio.
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)

        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(np.float32)

        sample_rate = TARGET_SAMPLE_RATE

    if not np.isfinite(audio).all():
        raise ValueError("Input audio contains invalid values.")

    return audio, sample_rate


# ==================================================
# ENHANCE AUDIO
# ==================================================

def enhance_audio(audio_path, enhanced_path=None):
    """
    Enhance a WAV file using MetricGAN+.

    Parameters
    ----------
    audio_path:
        Path to the original WAV file.

    enhanced_path:
        Optional output path supplied by main.py.
        If omitted, creates <filename>_enhanced.wav.

    Returns
    -------
    str:
        Path to the enhanced audio if successful;
        otherwise, path to the original audio.
    """

    input_path = Path(audio_path)

    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    input_path = input_path.resolve()

    # Validate the input file.
    if not input_path.is_file():
        logger.error("Audio file not found: %s", input_path)
        return str(input_path)

    if input_path.suffix.lower() != ".wav":
        logger.error("MetricGAN+ requires a WAV input file.")
        return str(input_path)

    # Determine the output path.
    if enhanced_path is not None:
        output_path = Path(enhanced_path)

        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path

        output_path = output_path.resolve()
    else:
        output_path = input_path.with_name(
            input_path.stem + "_enhanced.wav"
        )

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Read and preprocess the audio.
        audio, sample_rate = load_audio_file(input_path)

        # Avoid running the model on an empty signal.
        if len(audio) == 0:
            raise ValueError("No audio samples were found.")

        # 2. Convert NumPy audio to a tensor of shape [1, time].
        waveform = torch.from_numpy(
            np.ascontiguousarray(audio)
        ).unsqueeze(0)

        # 3. Load the model.
        enhancer = load_gan_model()

        # 4. Run MetricGAN+.
        logger.info(
            "Starting MetricGAN+ enhancement: %s",
            input_path.name,
        )

        with torch.no_grad():
            enhanced = enhancer.enhance_batch(
                waveform,
                lengths=torch.ones(1),
            )

        # 5. Convert the result to a NumPy array.
        if isinstance(enhanced, (tuple, list)):
            enhanced = enhanced[0]

        if not isinstance(enhanced, torch.Tensor):
            raise TypeError(
                "MetricGAN+ returned an unexpected output type."
            )

        enhanced_audio = (
            enhanced.detach()
            .cpu()
            .squeeze()
            .numpy()
        )

        enhanced_audio = np.asarray(
            enhanced_audio,
            dtype=np.float32,
        ).reshape(-1)

        if enhanced_audio.size == 0:
            raise ValueError(
                "MetricGAN+ returned empty enhanced audio."
            )

        if not np.isfinite(enhanced_audio).all():
            raise ValueError(
                "Enhanced audio contains invalid values."
            )

        # 6. Save the enhanced audio.
        sf.write(
            str(output_path),
            enhanced_audio,
            sample_rate,
            subtype="PCM_16",
        )

        # 7. Verify that the saved file is readable.
        saved_audio, saved_rate = sf.read(
            str(output_path),
            dtype="float32",
        )

        if saved_audio.size == 0 or saved_rate != sample_rate:
            raise ValueError(
                "The enhanced WAV file failed validation."
            )

        logger.info(
            "MetricGAN+ enhancement successful: %s",
            output_path,
        )

        return str(output_path)

    except Exception:
        logger.exception(
            "MetricGAN+ enhancement failed. "
            "Returning original audio: %s",
            input_path,
        )

        # Do not report enhancement as successful if it failed.
        return str(input_path)
