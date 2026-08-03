# Audio Overlap Removal

**English** | [简体中文](README.zh-CN.md)

Remove a known reference track from a mixture when the reference contains the
same source material but may follow a different timeline.

A typical use case is a livestream or screen recording:

- **C (mixture)**: presenter/streamer voice + game audio or BGM;
- **B (reference)**: a clean copy of the game audio or BGM;
- **A' (output)**: the result, with B removed while preserving the voice as
  much as possible.

This can be approximated as `C = A + B'`. B' contains the same material as B,
but may have undergone gain changes, lossy encoding, EQ, pauses, seeks,
replays, or slight playback-speed drift. The project first locates B within C
as a set of segments, then performs reference cancellation and protected
residual cleanup.

> This is not a general-purpose vocal or source separator. If the reference
> does not match the background in the mixture, low-confidence regions pass
> through unchanged; the algorithm cannot guess which sound should be removed.

## Features

- Automatically locates the reference track within the mixture;
- Reacquires after pauses, resumes, seeks, and replays;
- Tracks each segment's offset as a measured trajectory rather than one slope,
  and re-measures every chunk in a cheap low-rate pass before cancelling;
- Retries a chunk with a wider reference window instead of failing the run;
- Uses 250 ms local time warping, with a validated 16 ms fine path when useful;
- Decodes and processes in chunks instead of loading full-rate media at once;
- Supports mono, stereo, and multichannel inputs downmixed by FFmpeg;
- Preserves the mixture's mono/stereo layout;
- Scans and processes in parallel while preserving single-threaded output order;
- Delegates input decoding to FFmpeg and writes 24-bit FLAC or WAV.

The installed `audio-overlap-removal` command is the recommended interface.
Python projects can also import the `audio_overlap_removal` package directly.

## Installation

Requirements:

- Python 3.10 or later;
- `ffmpeg` and `ffprobe` available on `PATH`;
- Mixture and reference files no longer than 24 hours each;
- At least 8 GiB of memory recommended. Higher `--workers` values increase
  peak memory use.

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Verify that the external decoders are available:

```bash
ffmpeg -version
ffprobe -version
```

## Quick start

Scan and process the complete mixture:

```bash
audio-overlap-removal mixture.webm reference.webm clean.flac --strength 1
```

You can also run the module entry point from the source directory:

```bash
python -m audio_overlap_removal \
  mixture.webm reference.webm clean.flac \
  --strength 1
```

If the reference can only occur between 300 and 1200 seconds in the mixture,
limit scanning and processing to that range:

```bash
audio-overlap-removal \
  mixture.webm reference.webm clean.flac \
  --start 300 --end 1200 \
  --strength 1 --workers 4
```

`--start` and `--end` describe where the reference may occur on the mixture's
absolute timeline; they do not crop the output. The output always spans the
complete mixture. Content outside the range, unmatched content inside it, and
low-confidence regions all pass through unchanged. Only reliable matches are
reconstructed.

The output path must end in `.flac` or `.wav` and must differ from both input
paths.

**Start by listening at `--strength 1`.** Reduce it toward `0` if voice damage
is noticeable. If too much background remains, try `1.25–2`.

## Python API

Most callers only need the high-level `remove_reference()` function:

```python
from audio_overlap_removal import remove_reference

segments = remove_reference(
    "mixture.webm",
    "reference.webm",
    "clean.flac",
    start=300,
    end=1200,
    strength=1,
    workers=4,
)
```

The return value is a list of the `AlignmentSegment` objects that were actually
matched. The output still covers the complete mixture, and only those matched
regions are processed.

To separate scanning from processing:

```python
from audio_overlap_removal import process_audio, scan_reference

segments = scan_reference(
    "mixture.webm",
    "reference.webm",
    start=300,
    end=1200,
    workers=4,
)
process_audio(
    "mixture.webm",
    "reference.webm",
    "clean.wav",
    alignment_segments=segments,
    strength=1,
    workers=4,
)
```

`scan_reference()` only locates the reference; `process_audio()` only processes
the supplied matched regions. Names exported from the package root are the
stable public interface. Underscore-prefixed functions are internal and do not
carry compatibility guarantees.

The lower-level `fingerprint_media()`, `fingerprint_blocks()`, and
`FingerprintIndex` interfaces are also exported. Index candidates include a
`media_id` and a timestamp within that media, allowing multiple media files to
be indexed in memory. The current CLI builds a single-reference index and does
not persist a media library.

## Common options

| Option | Default | Description |
| --- | ---: | --- |
| `--start SECONDS` | `0` | Start of the range where the reference may occur |
| `--end SECONDS` | End of mixture | End of the range where the reference may occur |
| `--chunk SECONDS` | `30` | Processing chunk length |
| `--search SECONDS` | `0.25` | Reference search radius per chunk before a widened retry |
| `--workers N` | `4` | Number of parallel scanning/processing jobs |
| `--sample-rate HZ` | `48000` | Decode, processing, and output sample rate |
| `--strength VALUE` | `1` | Unified preservation/removal control |
| `--disable-adaptive-warp` | Off | Disable validated 16 ms fine time warping |
| `--disable-momentum` | Off | Skip the low-rate pass that measures each chunk's offset |

`--strength` has no hard upper limit:

| Value | Behavior |
| ---: | --- |
| `0` | Most conservative; protected reference cancellation only |
| `0–1` | Progressively adds residual and centered-media cleanup |
| `1` | Recommended starting point; balances removal and target protection |
| `1.25–2` | Favors media removal or ASR, with more risk to quiet speech |
| `>2` | Experimental; stronger cleanup with increasing damage risk |

`--cleanup-strength`, `--center-strength`, `--center-cleanup-strength`, and
`--silence-cleanup-strength` are expert overrides. Adjust `--strength` alone
first in most cases.

## Formats and channel layouts

### Input

The first audio stream is decoded through FFmpeg, so supported inputs normally
include:

- WAV, FLAC, and AIFF;
- MP3 and AAC/M4A;
- Ogg Vorbis and Opus;
- WebM and common video containers with audio;
- Other non-DRM formats supported by the installed FFmpeg build.

Sample rates and sample formats are converted to a common representation, so
the two inputs do not need to match. Corrupt files, encrypted or DRM-protected
media, codecs absent from the installed FFmpeg build, and files without an
audio stream are unsupported. Decode failures retain FFmpeg's error details.

### Channels

| Original input | Internal processing and output |
| --- | --- |
| Mono mixture | Processed and written as mono |
| Stereo mixture | Mid/Side processing, written as stereo |
| Mixture with more than 2 channels | Downmixed by FFmpeg and written as stereo |
| Mono reference | Mono/Mid reference path |
| Stereo reference | Mid/Side reference path |
| Reference with more than 2 channels | Downmixed by FFmpeg and used as stereo |

Multichannel downmixing discards the original surround layout. If 5.1 or 7.1
must be preserved, split and route the channels yourself; the current algorithm
only models mono and stereo.

### Output

There is no separate `--format` option. The output extension selects the format:

- `.flac`: FLAC container with 24-bit PCM;
- `.wav`: WAV container with 24-bit PCM.

Cover art, video, chapters, and other input metadata are not copied. The program
creates a hidden `.part` file in the output directory while writing, then
atomically replaces the target after a successful close. Failed writes remove
the partial file. Scanning and decoding do not create other temporary files on
disk.

## How it works

1. **Low-rate global scan**: inputs up to four hours use full correlation
   search to preserve established behavior; longer inputs build a compact
   streaming fingerprint index;
2. **Segment tracking and reacquisition**: rate-limited global searches run
   after local tracking fails, handling pauses, seeks, and replays. Each
   segment keeps the anchor trajectory it was measured from, so a long segment
   is not reduced to a single offset and slope;
3. **Offset momentum**: before any cancellation, every matched chunk is probed
   at 4 kHz over a wide window. The measurements are median-filtered into a
   continuous track, so an ambiguous chunk inherits its neighbours' offset
   instead of guessing;
4. **Local time alignment**: the chunk-local search starts from that prediction
   rather than scanning the whole window, then anchors define a time warp, with
   denser candidate paths validated when needed;
5. **Mid/Side reference cancellation**: the Side channel, which is less
   affected by centered speech, estimates a complex transfer function while
   the reference Mid helps process centered media;
6. **Protected residual cleanup**: presenter speech and unrelated stereo
   content are detected to automatically reduce later suppression;
7. **Per-chunk verification**: the correlation score, the momentum probe, and
   the energy the subtraction actually removed vote on whether to keep the
   cancelled chunk. A chunk that fails is retried once with a wider reference
   window and only then passes through;
8. **Chunked output**: chunks are written in timeline order with smoothed
   boundaries, and every low-confidence span is listed when the run finishes.

For deeper discussion of capabilities, boundaries, and experimental findings:

- [`docs/algorithm-review.md`](docs/algorithm-review.md)
- [`docs/real-world-goals.md`](docs/real-world-goals.md)

## Performance guidance

- Use `--start/--end` to reduce the scan range when the reference can only occur
  during part of the mixture;
- Start with `--workers 1` or `2` on memory-constrained systems;
- The default `--workers 4` is suitable when enough memory is available;
- High worker counts are usually limited by memory bandwidth and increase
  temporary memory roughly in proportion to concurrent chunks;
- The complete mixture is still reconstructed, so narrowing the scan range
  does not shorten the output.

When both the mixture scan range and reference are no longer than four hours,
the original full-correlation search is used to preserve common-case matching
behavior. If either is longer, the program automatically switches to the
streaming fingerprint index: FFmpeg audio blocks are piped directly into
feature generation without retaining the complete waveform or FFT and without
creating analysis files. At the 24-hour limit, the index and streaming windows
remain well below the total 8 GiB budget. Four workers primarily increase
concurrent cancellation-chunk memory; they do not create four copies of the
index.

## Tests

Run the complete regression suite:

```bash
python -m unittest -v test_audio_overlap_removal.py
```

Tests cover dynamic gain, pauses, seeks/replays, speed drift, short- and
long-media scan strategies, streaming FFmpeg decoding, multi-media fingerprint
IDs, mono/stereo routing, multichannel downmixing, WAV/FLAC selection, truncated
references, and deterministic parallel output order.

The long-media smoke test does not create multi-hour fixtures. Instead, a
96-second fixture compresses continuous playback, dynamic gain/EQ, pause and
resume, backward replay, and foreground-only regions, then forces an A/B
comparison between the FFT and fingerprint paths. The 24-hour capacity test
extrapolates from actual index bytes per window and asserts an upper bound.

## Known limitations

- The reference must derive from the same source as the target background;
  stylistic similarity is not enough;
- Heavy compression, remixes, complex dynamics processing, or large speed
  changes reduce matching and cancellation quality;
- The mono path has no safe Side control signal and is therefore more
  conservative than stereo;
- The case where both files are stereo but the background was downmixed to
  mono before mixing is not detected separately;
- Aggressive cleanup may damage unrelated, strongly centered background audio;
- Only the first audio stream from each input is selected;
- Surround layouts and media metadata are not preserved.

Low-confidence regions pass through unchanged to avoid incorrect cancellation.
If too much background remains, first verify that the reference content and
offset are correct, then increase `--strength` gradually.

## Project structure

```text
audio-overlap-removal/
├── audio_overlap_removal/
│   ├── alignment.py       # Reference scanning and timeline matching
│   ├── cancellation.py    # Signal alignment, cancellation, and cleanup
│   ├── fingerprint.py     # Streaming fingerprints and candidate index
│   ├── media.py           # FFmpeg decoding, probing, and atomic output
│   ├── models.py          # Public data models and strength profiles
│   ├── parallel.py        # Bounded, order-preserving parallel execution
│   ├── pipeline.py        # Independently callable high-level pipeline
│   └── cli.py             # Command-line options and entry point
├── test_audio_overlap_removal.py  # Algorithm, API, and I/O regressions
├── pyproject.toml                 # Package metadata, dependencies, and CLI
└── docs/                          # Goals, experiments, and algorithm review
```
