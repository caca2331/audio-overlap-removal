"""Known-reference audio removal library."""

from .alignment import discover_alignment_segments
from .models import AlignmentSegment, CancellationProfile, cancellation_profile
from .pipeline import process_audio, remove_reference, scan_reference

__all__ = [
    "AlignmentSegment",
    "CancellationProfile",
    "cancellation_profile",
    "discover_alignment_segments",
    "process_audio",
    "remove_reference",
    "scan_reference",
]
