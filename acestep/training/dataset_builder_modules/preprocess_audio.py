import numpy as np
import torch


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate.

    Uses librosa (FFmpeg under the hood) instead of torchaudio.load to avoid
    torchcodec's CUDA-version coupling — torchcodec wheels link against a
    specific CUDA major (e.g. 13) and dlopen libnvrtc/libnppi at runtime,
    which fails whenever the installed torch is on a different CUDA build
    (e.g. cu128). Librosa just shells out to FFmpeg and returns numpy.
    """
    import librosa  # imported lazily so non-preprocess code paths don't pay the cost

    audio_np, _sr = librosa.load(
        audio_path,
        sr=target_sample_rate,
        mono=False,
        duration=max_duration,
    )

    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np], axis=0)
    elif audio_np.shape[0] == 1:
        audio_np = np.repeat(audio_np, 2, axis=0)
    elif audio_np.shape[0] > 2:
        audio_np = audio_np[:2, :]

    audio = torch.from_numpy(audio_np).float()
    return audio, target_sample_rate
