"""Known-reference audio removal library."""

import logging

from ._version import __version__
from .alignment import discover_alignment_segments
from .fingerprint import (
    FingerprintCandidate,
    FingerprintIndex,
    FingerprintTrack,
    fingerprint_blocks,
    fingerprint_media,
)
from .models import AlignmentSegment, CancellationProfile, cancellation_profile
from .pipeline import process_audio, remove_reference, scan_reference

# Progress and diagnostics go through the logging hierarchy. A library must
# not decide where they land, so nothing is emitted until an application adds
# a handler of its own; the command-line entry point installs one.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AlignmentSegment",
    "CancellationProfile",
    "FingerprintCandidate",
    "FingerprintIndex",
    "FingerprintTrack",
    "__version__",
    "cancellation_profile",
    "discover_alignment_segments",
    "fingerprint_blocks",
    "fingerprint_media",
    "process_audio",
    "remove_reference",
    "scan_reference",
]
