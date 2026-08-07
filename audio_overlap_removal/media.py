"""FFmpeg probing/decoding and atomic audio output."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 48_000
MIN_SAMPLE_RATE = 1_000
MIN_ALIGNMENT_SAMPLE_RATE = 200
MAX_MEDIA_DURATION_SEC = 24.0 * 60.0 * 60.0
FFMPEG_DIR_ENV = "AOR_FFMPEG_DIR"
_OUTPUT_FORMATS = {
    ".flac": ("FLAC", "PCM_24"),
    ".wav": ("WAV", "PCM_24"),
}


def _frozen_dir() -> Path | None:
    """The directory of a packaged executable, or None when run from source."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return Path(sys.executable).resolve().parent
    return None


def _preferred_tool_dirs() -> list[Path]:
    """Locations that outrank PATH because the user chose them explicitly."""
    dirs = []
    override = os.environ.get(FFMPEG_DIR_ENV)
    if override:
        dirs.append(Path(override))
    frozen = _frozen_dir()
    if frozen is not None:
        # A packaged build has no shell profile to edit, so dropping FFmpeg
        # beside the executable has to be a supported way to install it.
        dirs.extend([frozen, frozen / "bin"])
    return dirs


def _fallback_tool_dirs() -> list[Path]:
    """Common install locations, tried only when PATH has nothing."""
    home = Path.home()
    if sys.platform == "win32":
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_data = os.environ.get("ProgramData", r"C:\ProgramData")
        local_app_data = os.environ.get("LOCALAPPDATA", str(home / "AppData/Local"))
        return [
            Path(program_files) / "ffmpeg/bin",
            Path(r"C:\ffmpeg\bin"),
            Path(program_data) / "chocolatey/bin",
            home / "scoop/shims",
            Path(local_app_data) / "Microsoft/WinGet/Links",
        ]
    if sys.platform == "darwin":
        return [
            Path("/opt/homebrew/bin"),
            Path("/usr/local/bin"),
            Path("/opt/local/bin"),
            home / ".local/bin",
        ]
    return [
        Path("/usr/local/bin"),
        Path("/snap/bin"),
        Path("/home/linuxbrew/.linuxbrew/bin"),
        home / ".local/bin",
    ]


@lru_cache(maxsize=None)
def _tool(name: str) -> str:
    """Resolve an FFmpeg tool to an absolute path, or to its bare name.

    Returning the bare name when nothing matches keeps the not-found error in
    one place: the subprocess call that was going to run it.
    """
    exe = f"{name}.exe" if os.name == "nt" else name

    def first_match(directories: list[Path]) -> str | None:
        for directory in directories:
            candidate = directory / exe
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return None

    # PATH sits in the middle: an explicit choice beats it, and the guessed
    # install locations must never override a deliberately configured PATH.
    return (
        first_match(_preferred_tool_dirs())
        or shutil.which(exe)
        or first_match(_fallback_tool_dirs())
        or exe
    )


def _child_env() -> dict[str, str] | None:
    """Environment for FFmpeg, or None to inherit this process's.

    A packaged build runs with a loader path pointing at its own bundled
    libraries. Children inherit it, so a system FFmpeg would resolve
    libstdc++ and friends against the versions the bundle was built with and
    fail to start. PyInstaller stashes any pre-launch value in *_ORIG.
    """
    if _frozen_dir() is None:
        return None
    env = os.environ.copy()
    for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        original = env.pop(f"{name}_ORIG", None)
        if original is None:
            env.pop(name, None)
        else:
            env[name] = original
    return env


def _missing_tool_message(command: str) -> str:
    name = Path(command).stem
    return (
        f"{name} was not found. Install FFmpeg so that ffmpeg and ffprobe are "
        f"on PATH, or set {FFMPEG_DIR_ENV} to the directory that contains them."
    )


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
            env=_child_env(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(_missing_tool_message(command[0])) from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        detail = (stderr or "").strip()
        # The command carries a resolved absolute path; report the tool name.
        message = f"{Path(command[0]).stem} failed with exit code {error.returncode}"
        if detail:
            message += f": {detail}"
        raise RuntimeError(message) from error


def _probe_audio(path: str) -> dict:
    media = Path(path)
    if not media.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    command = [
        _tool("ffprobe"),
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
    blocks = list(
        _iter_decode_mono_low(
            path,
            sr,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
    )
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)


def _iter_decode_mono_low(
    path: str,
    sr: int,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    *,
    block_frames: int = 65_536,
) -> Iterator[np.ndarray]:
    """Yield mono float32 blocks directly from FFmpeg's stdout pipe."""
    if sr < MIN_ALIGNMENT_SAMPLE_RATE:
        raise ValueError(
            f"decode sample rate must be at least {MIN_ALIGNMENT_SAMPLE_RATE} Hz."
        )
    if not np.isfinite(start_sec) or start_sec < 0.0:
        raise ValueError("decode start must be non-negative and finite.")
    if duration_sec is not None and (
        not np.isfinite(duration_sec) or duration_sec <= 0.0
    ):
        raise ValueError("decode duration must be positive and finite.")
    if block_frames < 1:
        raise ValueError("block_frames must be positive.")
    command = [
        _tool("ffmpeg"),
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
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_child_env(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(_missing_tool_message(command[0])) from error

    stderr_tail = bytearray()

    def drain_stderr() -> None:
        if process.stderr is None:
            return
        while chunk := process.stderr.read(8_192):
            stderr_tail.extend(chunk)
            if len(stderr_tail) > 65_536:
                del stderr_tail[:-65_536]

    stderr_thread = threading.Thread(
        target=drain_stderr,
        name="ffmpeg-stderr",
        daemon=True,
    )
    stderr_thread.start()
    try:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Could not open FFmpeg streaming pipes.")
        sample_bytes = np.dtype("<f4").itemsize
        byte_count = block_frames * sample_bytes
        carry = b""
        while block := process.stdout.read(byte_count):
            block = carry + block
            usable = len(block) - len(block) % sample_bytes
            if usable:
                yield np.frombuffer(block[:usable], dtype="<f4").copy()
            carry = block[usable:]
        if carry:
            raise RuntimeError("FFmpeg returned a truncated float32 sample.")
        return_code = process.wait()
        stderr_thread.join()
        if return_code:
            stderr = stderr_tail.decode(errors="replace").strip()
            message = f"ffmpeg failed with exit code {return_code}"
            if stderr:
                message += f": {stderr}"
            raise RuntimeError(message)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        stderr_thread.join()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


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
        _tool("ffmpeg"),
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
