"""Known-reference audio removal library."""

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

__all__ = [
    "AlignmentSegment",
    "CancellationProfile",
    "FingerprintCandidate",
    "FingerprintIndex",
    "FingerprintTrack",
    "cancellation_profile",
    "discover_alignment_segments",
    "fingerprint_blocks",
    "fingerprint_media",
    "process_audio",
    "remove_reference",
    "scan_reference",
]
