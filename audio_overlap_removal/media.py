"""FFmpeg probing/decoding and atomic audio output."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 48_000
MIN_SAMPLE_RATE = 1_000
MIN_ALIGNMENT_SAMPLE_RATE = 200
_OUTPUT_FORMATS = {
    ".flac": ("FLAC", "PCM_24"),
    ".wav": ("WAV", "PCM_24"),
}


def _run_media_command(
    command: list[str], *, text: bool = False
) -> subprocess.CompletedProcess:
    """Run FFmpeg/FFprobe and preserve the useful diagnostic on failure."""
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=text,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            f"{command[0]} was not found. Install FFmpeg and ensure both "
            "ffmpeg and ffprobe are available on PATH."
        ) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        detail = (stderr or "").strip()
        message = f"{command[0]} failed with exit code {error.returncode}"
        if detail:
            message += f": {detail}"
        raise RuntimeError(message) from error


def _probe_audio(path: str) -> dict:
    media = Path(path)
    if not media.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,duration:format=duration",
        "-of",
        "json",
        str(media),
    ]
    result = _run_media_command(command, text=True)
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"ffprobe returned invalid metadata for {path!r}."
        ) from error
    streams = payload.get("streams") or []
    if not streams:
        raise ValueError(f"No audio stream was found in {path!r}.")
    return payload


def _media_duration(path: str) -> float:
    payload = _probe_audio(path)
    candidates = [
        (payload.get("format") or {}).get("duration"),
        (payload.get("streams") or [{}])[0].get("duration"),
    ]
    for candidate in candidates:
        try:
            duration = float(candidate)
        except (TypeError, ValueError):
            continue
        if np.isfinite(duration) and duration > 0.0:
            return duration
    raise ValueError(f"Could not determine the duration of {path!r}.")


def _audio_channel_count(path: str) -> int:
    payload = _probe_audio(path)
    try:
        channels = int(payload["streams"][0]["channels"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Could not determine the audio channel count of {path!r}."
        ) from error
    if channels < 1:
        raise ValueError(f"Invalid audio channel count in {path!r}: {channels}.")
    return channels


def _processing_channel_count(native_channels: int) -> int:
    """Keep mono intact and let FFmpeg downmix every wider layout to stereo."""
    return 1 if native_channels == 1 else 2


def _output_settings(path: Path) -> tuple[str, str]:
    try:
        return _OUTPUT_FORMATS[path.suffix.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(_OUTPUT_FORMATS))
        raise ValueError(
            f"Unsupported output extension {path.suffix or '<none>'!r}; "
            f"use one of: {supported}."
        ) from error


def _paths_refer_to_same_file(first: str | Path, second: str | Path) -> bool:
    left = Path(first)
    right = Path(second)
    try:
        if left.exists() and right.exists() and os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return left.resolve(strict=False) == right.resolve(strict=False)


@contextmanager
def _atomic_soundfile(
    output: Path,
    *,
    samplerate: int,
    channels: int,
    format: str,
    subtype: str,
):
    """Replace the destination only after the temporary output closes cleanly."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".part",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with sf.SoundFile(
            temporary,
            mode="w",
            samplerate=samplerate,
            channels=channels,
            format=format,
            subtype=subtype,
        ) as sink:
            yield sink
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_mono_low(
    path: str,
    sr: int,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> np.ndarray:
    if sr < MIN_ALIGNMENT_SAMPLE_RATE:
        raise ValueError(
            f"decode sample rate must be at least {MIN_ALIGNMENT_SAMPLE_RATE} Hz."
        )
    if start_sec < 0.0:
        raise ValueError("decode start must be non-negative.")
    if duration_sec is not None and duration_sec <= 0.0:
        raise ValueError("decode duration must be positive.")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
    ]
    if start_sec > 0.0:
        command.extend(["-ss", f"{start_sec:.9f}"])
    command.extend(
        [
            "-i",
            path,
            "-map",
            "0:a:0",
        ]
    )
    if duration_sec is not None:
        command.extend(["-t", f"{duration_sec:.9f}"])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    result = _run_media_command(command)
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def _decode_stereo(
    path: str,
    start_sec: float,
    duration_sec: float,
    sr: int = DEFAULT_SR,
    source_channels: int = 2,
) -> np.ndarray:
    if sr < MIN_SAMPLE_RATE:
        raise ValueError(f"sample rate must be at least {MIN_SAMPLE_RATE} Hz.")
    if start_sec < 0.0:
        raise ValueError("decode start must be non-negative.")
    if duration_sec <= 0.0:
        raise ValueError("decode duration must be positive.")
    if source_channels not in (1, 2):
        raise ValueError("source_channels must be 1 or 2.")
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{start_sec:.9f}",
        "-i",
        path,
        "-map",
        "0:a:0",
        "-t",
        f"{duration_sec:.9f}",
        "-vn",
        "-ac",
        str(source_channels),
        "-ar",
        str(sr),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = _run_media_command(command)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    remainder = len(audio) % source_channels
    if remainder:
        audio = audio[:-remainder]
    audio = audio.reshape(-1, source_channels)
    if source_channels == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio
