# Two-pass residual-reference exploration

This tracked research directory contains offline prototypes and their generated
metric snapshots. Production source code is intentionally unchanged. Generated
FLAC listening samples remain ignored.

The experiments expect `miyako.webm`, `sr.webm`, and `ya.mp3` in the directory
specified by `AUDIO_OVERLAP_ASSETS`. If the variable is omitted, the scripts
use the original sibling-workspace location:
`../Agent-Task-Playground/audio-removal/extra-asset`.

Run scripts from this directory with the project environment active. The main
sequence is:

1. `python explore_two_pass.py`
2. `python explore_iterative.py`
3. `python test_exact_pipeline.py`
4. `python explore_dual_model.py`
5. `python test_dual_high_floor.py`

The `export_*.py` scripts regenerate optional real-material listening samples.
See `docs/iterative-cancellation-exploration.md` for the chronological
reasoning, including rejected hypotheses.

## Listening order

`all-comparisons.flac` contains three 12-second groups (`early`, `middle`,
`late`). Each group is:

1. original mixture;
2. current `strength=1` output;
3. proposed two-pass output.

There is one second of silence between items and two seconds between groups.
The individual files use the same labels.

The middle clip is the useful discriminator. The second pass is effectively
inactive in the early and late clips.

## Prototype

The first correction estimates a remaining reference transfer from the
foreground-safer Side residual, applies it to reference Mid, and caps it to 25%
of the first-pass prediction. A strict Mid fallback was tested but did not
materially help and is not part of the selected variant.

The second experimental correction fits persistent odd nonlinear terms from
Side (for soft clipping). It is cross-validated on alternating one-second
blocks. This branch did not activate on the three real clips.

The correction is computed from the linear residual, before interpreting any
spectral cleanup as removed media. For the listening prototype only, the
bounded correction is applied to the current full output. A production
implementation should insert it before cleanup and recompute the cleanup mask.

## Results

- Synthetic moving-gain case: 25.91 -> 28.10 dB suppression, with foreground
  coefficient unchanged at 0.00 dB.
- Synthetic soft clipping: 14.51 -> 14.65 dB on the complete strength-1 output.
- Fixed FIR, path switch, asymmetric L/R path, stereo double-talk, voice-only,
  and unrelated-music-only cases showed no measurable regression.
- Real middle clip: correction RMS is 0.99% of current-output RMS; the
  conservative reference-related residual proxy falls by about 11.8%.
- Real early and late clips: correction is effectively zero.

These results support a small, safely gated improvement for time-varying
playback gain. They do not support treating the second pass as a general
solution for every residual type.

## Iterative follow-up

`explore_iterative.py` compares the current linear (`strength=0`) model, a
Side-only conservative initial model, and up to six repeated Side corrections.

- The conservative initial model keeps the synthetic foreground coefficient
  near 0 dB, versus roughly -0.06 to -0.13 dB for the current linear model.
- That safety costs substantial cancellation quality for path switches,
  asymmetric L/R transfer, and soft clipping, so it is not a wholesale
  replacement for the current model.
- Starting from the current `strength=0` model, the moving-gain case progresses
  25.91 -> 28.10 -> 28.22 -> 28.25 dB over the first three corrections.
  Other scenarios remain effectively unchanged.
- On the real middle clip, Side-residual improvement per iteration falls from
  0.61% to 0.20%, 0.11%, 0.08%, 0.06%, and 0.04%.

The practical schedule is therefore a model pass with cleanup disabled,
followed by at most two Side-residual corrections and then the normal final
cleanup. A convergence threshold should normally stop after one pass and
occasionally allow the second. Unbounded iteration is not justified by these
results.

## Exact end-to-end validation

`test_exact_pipeline.py` tests the proposed ordering exactly: linear prediction
with cleanup disabled, zero to two gated Side corrections, then one cleanup
computed from the corrected prediction. A rejected loop returns through the
existing path so non-triggered cases retain the current output.

- Of nine synthetic cases, only moving gain triggered: 25.91 -> 28.22 dB,
  with foreground unchanged at 0.00 dB.
- All other synthetic cases triggered zero iterations and exactly retained the
  current result.
- Real early and late clips triggered zero iterations.
- Real middle triggered two iterations. After recomputing the final cleanup,
  output difference RMS was 0.43% of the current output and the conservative
  reference-related residual proxy improved by only about 0.59%.

The earlier 11.8% real proxy improvement came from approximately adding the
correction to an already-cleaned output. The exact pipeline shows that cleanup
mask recomputation absorbs most of that apparent gain. The structure is safe
under the tested gate, but its real-world benefit is too small to justify
integration on this evidence alone.

## B_safe / B_full dual model

`explore_dual_model.py` computes a Side-only `B_safe` and the current linear
`B_full`. Their difference is treated as a risky Mid-derived prediction. Low
retention floors failed badly on real material because the Mid-specific path is
not a small tail: `B_full - B_safe` is 39% to 59% of the full prediction RMS.

A 90% minimum retention floor was the useful compromise:

- Eleven synthetic truth scenarios all held or improved. Examples:
  path switch 17.91 -> 18.15 dB, side double-talk 18.12 -> 18.32 dB, and
  asymmetric L/R 13.74 -> 13.80 dB.
- Foreground coefficients improved slightly or remained unchanged.
- Real output difference was 0.82%, 1.12%, and 1.82% RMS for early, middle,
  and late.
- The conservative reference-related residual proxy improved by 9.2%, 0.5%,
  and 2.2%, respectively. The early absolute residual was already extremely
  small, so its large relative percentage should not be interpreted as an
  audible change.

The resulting hybrid candidate is: retain the current digital-mix fast path
with its gated Side loop; otherwise use the 90%-floor dual model and then one
normal cleanup. This is more promising than unrestricted repeated passes, but
the real improvements remain small.
