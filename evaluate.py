import os
import sys
import time
import librosa
import numpy as np
from main import process_audio, SR

def calculate_a_similarity(ref, test):
    if ref is None: return None
    n = min(len(ref), len(test))
    if n < 100: return 0.0
    a, b = ref[:n] - np.mean(ref[:n]), test[:n] - np.mean(test[:n])
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / d) if d > 0 else 0.0

def calculate_b_residual(y_out, y_b_aligned):
    n = min(len(y_out), len(y_b_aligned))
    a, b = y_out[:n], y_b_aligned[:n]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(abs(np.sum(a * b) / (d + 1e-10)))

def calculate_sam(y_out):
    S = np.abs(librosa.stft(y_out, n_fft=4096))
    # Focus on high frequencies where musical noise is most prominent
    n_bins = S.shape[0]
    high_S = S[n_bins//4:] # Top 75% of frequencies
    diffs = np.abs(np.diff(high_S, axis=0))
    jaggedness = np.mean(diffs) / (np.mean(high_S) + 1e-10)
    return float(jaggedness)

def calculate_removal_quality(y_mix, y_out, sr=SR, clip_sec=2.0):
    n = min(len(y_mix), len(y_out))
    y_m, y_o = y_mix[:n], y_out[:n]
    diff_ratio = np.sqrt(np.mean((y_m - y_o)**2)) / (np.sqrt(np.mean(y_m**2)) + 1e-10)
    
    clip_s = int(clip_sec * sr)
    n_clips = n // clip_s
    reductions = []
    for i in range(n_clips):
        rms_m = np.sqrt(np.mean(y_m[i*clip_s:(i+1)*clip_s]**2) + 1e-10)
        rms_o = np.sqrt(np.mean(y_o[i*clip_s:(i+1)*clip_s]**2) + 1e-10)
        reductions.append(20 * np.log10(rms_m / rms_o))
    
    reductions = np.array(reductions)
    score = diff_ratio * min(np.mean(reductions >= 0.1) / 0.5, 1.0) if n_clips > 0 else diff_ratio
    return {"diff_ratio": diff_ratio, "median_db": np.median(reductions) if n_clips > 0 else 0.0,
            "coverage_05": np.mean(reductions >= 0.5) if n_clips > 0 else 0.0,
            "score": score, "passed": diff_ratio >= 0.05, "sam": calculate_sam(y_o)}

TEST_DESCRIPTIONS = {
    1: "Standard Mix", 2: "Unstable Volume B", 3: "Abrupt Volume Change", 4: "Delayed Start",
    5: "Early End", 6: "B Given is Longer", 7: "MP3 Compression", 8: "Sample Rate Mismatch",
    9: "Loud BGM", 10: "EQ / Filter", 11: "Audio Ducking", 12: "Streamer Pause/Gap",
    13: "Stream Skip", 14: "Network Stutter", 15: "90-Min Real World", 16: "T15S: Real World Short"
}

TEST_FILES = { 16: ("C15S.flac", "B15S.flac") }

def main():
    ref_file = 'extra-test-out/base_a.flac'
    y_ref, _ = librosa.load(ref_file, sr=SR) if os.path.exists(ref_file) else (None, None)
    
    to_run = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else sorted(TEST_DESCRIPTIONS.keys())
    results = {}

    for i in to_run:
        c_fn, b_fn = TEST_FILES[i] if i in TEST_FILES else (f"C{i}.flac", f"B{i}.flac")
        m_p, b_p, o_p = f"extra-test-out/{c_fn}", f"extra-test-out/{b_fn}", f"extra-test-out/out_{c_fn}"
        if not os.path.exists(m_p): continue

        print(f"Running T{i}...")
        t0 = time.time()
        y_out, y_b_aligned = process_audio(m_p, b_p, o_p)
        y_mix, _ = librosa.load(m_p, sr=SR)
        
        sim = calculate_a_similarity(y_ref, y_out) if i < 15 else None
        res = calculate_b_residual(y_out, y_b_aligned)
        q = calculate_removal_quality(y_mix, y_out)
        results[i] = {"sim": sim, "res": res, "q": q, "time": time.time() - t0}

    print("\n=== Results ===")
    hdr = f"{'Test':<5} | {'Description':<22} | {'A-Sim':>6} | {'B-Res':>6} | {'DiffR':>5} | {'SAM':>6} | {'Gate'}"
    print(hdr)
    print("-" * len(hdr))
    for i in sorted(results):
        r = results[i]
        sim = f"{r['sim']:.3f}" if r['sim'] is not None else "N/A"
        print(f"{i:<5} | {TEST_DESCRIPTIONS[i]:<22} | {sim:>6} | {r['res']:.3f} | {r['q']['diff_ratio']:.3f} | {r['q']['sam']:>6.2f} | {'PASS' if r['q']['passed'] else 'FAIL'}")

if __name__ == '__main__':
    main()
