# Changelog

## 0.2.0 — 2026-09-13

Chunk alignment turned out to be the first bottleneck on real material, not
the transfer model. A ground-truth benchmark (`experiments/artifact-bench`)
that pushes a real reference window through a synthetic broadcast chain and
mixes it into real host audio locates the residual per degradation stage;
the numbers below are from it unless stated otherwise.

### Cancellation

- 250 ms alignment anchors are measured at the playback rate the offset track
  implies, and re-measured when the anchors disagree with that rate. A steady
  0.1% speed difference used to smear each anchor by 2–4 samples and cost
  10–13 dB; the benchmark's drift-only case goes from 9.9/12.9 dB to
  22.8/26.2 dB, the same as with no degradation at all.
- Anchors that jump more than 1 ms away from their neighbours or from the
  robust offset line (one period of the music away, typically on a quiet or
  tonal passage) are measured again next to the line and dropped if the
  near-line peak is not credible. A mild playback EQ used to cost 7 dB this
  way; the EQ-only case goes from 15.0/18.9 dB to 22.6/25.0 dB. A seek inside
  a chunk is told apart from a wrong period and its anchors are kept.
- The aligned reference is rebuilt with a cubic spline instead of linear
  interpolation, whose fraction-dependent low-pass capped cancellation near
  23 dB on fractional positions.
- The Mid fallback prediction in bins where Side has no energy carries a
  chunk-level static transfer read from Mid instead of a bare scalar gain.
- Consecutive cancelled chunks fade into each other across their shared
  context instead of a hard cut bridged by a one-sample correction ramp.
- Complete broadcast chain (EQ, balance, drift, gain, Opus), `strength 0`:
  8.1/14.1 dB before, 18.6/19.2 dB after. On the real reference material the
  measured reduction rises by a few tenths of a decibel overall and by 2–3 dB
  on the chunks whose anchors had been wrong; no chunk gets worse by more
  than 0.2 dB.

### Matching

- After a reacquisition both scans walk back over the local steps skipped
  while waiting for it, so a replay starts where it really started and a
  replay shorter than the reacquisition delay still forms a segment. On the
  real material this recovers a 58 s replay the old scan missed entirely.

### Robustness and diagnostics

- A chunk whose cancellation raises (an FFmpeg failure, a numerical edge
  case) is logged with its traceback and written unmodified; one chunk can no
  longer end a run that is hours long.
- `process_audio` rejects `search_sec=0` up front instead of failing inside
  the first chunk; the result summary aggregates only cancelled chunks.
- The run result records the playback rate prior and the rate the anchors
  settled on, the anchor count, and the Side residual per band, which tells
  codec damage in the top bands from model error below.
- The log and result documents introduced after 0.1.0: progress goes to
  stderr and a per-line-flushed log file, and a single JSON result document
  records settings, inputs, segments, every chunk and a summary, written
  even when a run is interrupted.

### Performance

- The fingerprint index searches with a dot product instead of a k-d tree,
  which at 512 dimensions was 15 times slower and held a float64 copy of the
  signatures: a 24-hour index needs about 0.36 GB instead of 1.4 GB.
- The canceller streams fewer arrays per chunk; the traced peak of one 32 s
  stereo chunk drops from 806 MB to 535 MB with unchanged output.

### Documentation

- `docs/algorithm-review.md` records the benchmark decomposition, each change
  with its measured effect, the STFT window A/B (2048/512 stays), the
  scan-cost analysis explaining why global reacquisition was left alone, and
  corrects the earlier verdict that the 4645–4663 s fingerprint match was a
  false positive.

## 0.1.0 — 2026-08-07

First tagged release: chunked scan and cancellation with Mid/Side transfer
functions, protected residual cleanup, offset momentum, per-chunk voting,
fingerprint indexing for long media, and standalone Windows and macOS builds.
