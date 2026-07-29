import os
import sys
import time
import numpy as np
import scipy.signal
import scipy.interpolate
import scipy.ndimage
import librosa
import soundfile as sf

SR = 44100
STFT_N_FFT = 4096
STFT_HOP = 1024

def _estimate_scale(mag_a, mag_b):
    n_bins, n_frames = mag_a.shape
    scales = np.ones((n_bins, n_frames), dtype=np.float32)
    band_edges = np.linspace(0, n_bins, 33).astype(int)
    freqs = np.linspace(0, SR/2, n_bins)
    for b in range(len(band_edges)-1):
        s, e = band_edges[b], band_edges[b+1]
        ba, bb = mag_a[s:e], mag_b[s:e]
        mid_freq = freqs[s]
        hop = 10
        for t in range(0, n_frames, hop):
            et = min(t + hop, n_frames)
            bba, bbb = ba[:, t:et], bb[:, t:et]
            mask = bbb > np.percentile(bbb, 80)
            if mask.sum() > 5:
                ratios = bba[mask] / (bbb[mask] + 1e-10)
                # Use 70th percentile for better coverage of bg peaks
                sc = np.percentile(ratios, 70)
                scales[s:e, t:et] = np.clip(sc, 0.01, 20.0)
            elif t > 0: scales[s:e, t:et] = scales[s:e, max(0, t-1):t]
    
    # Smooth scales in time and frequency
    scales = scipy.ndimage.median_filter(scales, size=(1, 5))
    scales = scipy.ndimage.gaussian_filter(scales, sigma=(1.0, 2.0))
    return scales

def _run_alignment(ya, yb, sr):
    # Coarse
    ya_lo = librosa.resample(ya, orig_sr=sr, target_sr=2000)
    yb_lo = librosa.resample(yb, orig_sr=sr, target_sr=2000)
    corr = scipy.signal.fftconvolve(yb_lo, ya_lo[::-1], mode='full')
    lags = scipy.signal.correlation_lags(len(yb_lo), len(ya_lo))
    off = -lags[np.argmax(corr)] / 2000.0
    print(f"  Probe offset: {off:.2f}s")
    
    # Piecewise
    hop_sec = 0.5 
    win_sec = 8.0
    sr_al = 16000 # Higher resolution for sample-level alignment
    ya_al = librosa.resample(ya, orig_sr=sr, target_sr=sr_al)
    yb_al = librosa.resample(yb, orig_sr=sr, target_sr=sr_al)
    
    traj = []
    for t in np.arange(0, len(ya)/sr - win_sec, hop_sec):
        s_a = int(t * sr_al)
        e_a = s_a + int(win_sec * sr_al)
        chunk_a = ya_al[s_a:e_a]
        
        search_win = 30.0 # Large window for skips
        s_b_exp = t - off
        s_b = int(max(0, s_b_exp - search_win) * sr_al)
        e_b = int(min(len(yb_al), s_b_exp + win_sec + search_win) * sr_al)
        search_b = yb_al[s_b:e_b]
        
        if len(search_b) > len(chunk_a):
            c = scipy.signal.fftconvolve(search_b, chunk_a[::-1], mode='valid')
            best = np.argmax(c)
            denom = np.sqrt(np.sum(search_b[best:best+len(chunk_a)]**2) * np.sum(chunk_a**2)) + 1e-10
            corr_val = c[best] / denom
            
            curr_off = t - (s_b / sr_al + best / sr_al)
            # More aggressive jump following
            if corr_val > 0.3 or abs(curr_off - off) < 2.0:
                off = curr_off
        traj.append((t, off))
    
    y_ba = np.zeros_like(ya)
    ft = np.arange(len(ya)) / sr
    st, sv = [x[0] for x in traj], [x[1] for x in traj]
    # Median filter to remove outliers
    sv = scipy.signal.medfilt(sv, kernel_size=3)
    f = scipy.interpolate.interp1d(st, sv, kind='cubic', fill_value='extrapolate')
    offs = f(ft)
    src = (ft - offs) * sr
    mask = (src >= 0) & (src < len(yb)-1)
    idx = src[mask].astype(np.int32)
    fr = src[mask] - idx
    y_ba[mask] = (1-fr)*yb[idx] + fr*yb[idx+1]
    return y_ba

def process_audio(m_p, b_p, o_p):
    y_a, _ = librosa.load(m_p, sr=SR)
    y_b, _ = librosa.load(b_p, sr=SR)
    y_ba = _run_alignment(y_a, y_b, SR)
    # Block-wise processing with Overlap-Add (OLA)
    y_out = np.zeros_like(y_a)
    y_norm = np.zeros_like(y_a)
    bs = 60 * SR
    hop = bs // 2 # 50% overlap
    pad = 2 * SR
    
    # Block window for smooth blending
    win = np.hanning(bs)
    
    # Previous block gain for Decision-Directed SNR estimation
    prev_g_ma = None
    
    for s in range(0, len(y_a), hop):
        e = min(s + bs, len(y_a))
        w = win if (e - s == bs) else np.hanning(e - s)
        
        ps, pe = max(0, s-pad), min(len(y_a), e+pad)
        ma_full = librosa.stft(y_a[ps:pe], n_fft=STFT_N_FFT, hop_length=STFT_HOP)
        ma, pa = np.abs(ma_full), np.angle(ma_full)
        mb = np.abs(librosa.stft(y_ba[ps:pe], n_fft=STFT_N_FFT, hop_length=STFT_HOP))
        
        sc = _estimate_scale(ma, mb)
        # Power of noise (background)
        pn = (mb * sc)**2
        # Power of mixed signal
        pc = ma**2
        
        # Decision-Directed SNR Estimation
        # G_DD = alpha * (prev_P_A / P_N) + (1-alpha) * max(P_C/P_N - 1, 0)
        # We use a simpler version: estimate prior SNR xi
        xi = np.maximum(pc / (pn + 1e-10) - 1.0, 0.0)
        xi_smooth = scipy.ndimage.gaussian_filter(xi, sigma=(1.5, 0.5))
            
        # Wiener Gain with adaptive over-subtraction
        # Higher alpha_sub in noise, lower in speech
        alpha_sub = 0.3 + 2.2 * np.exp(-0.5 * xi_smooth)
        gain = xi / (xi + alpha_sub)
        
        # Soft Noise Gate: Only apply in low SNR regions
        gate_mask = np.where(xi_smooth < 0.2, 1.0, 0.0)
        gain = np.where(gate_mask > 0.5, gain**1.5, gain)
        gain = np.sqrt(np.maximum(gain, 0.005))
        
        # Frequency-dependent smoothing and artifact rejection
        gain = scipy.ndimage.median_filter(gain, size=(3, 1))
        gain = scipy.ndimage.gaussian_filter(gain, sigma=(1.0, 0.5))
        
        blk = librosa.istft((ma*gain)*np.exp(1j*pa), length=pe-ps, n_fft=STFT_N_FFT, hop_length=STFT_HOP)
        
        # Smooth taper at block edges to prevent clicks
        taper = np.ones(pe-ps)
        taper_len = int(0.1 * SR)
        if ps > 0: taper[:taper_len] *= np.linspace(0, 1, taper_len)
        if pe < len(y_a): taper[-taper_len:] *= np.linspace(1, 0, taper_len)
        blk *= taper
        
        chunk = blk[s-ps : s-ps+e-s]
        
        y_out[s:e] += chunk * w
        y_norm[s:e] += w
    
    y_out /= (y_norm + 1e-10)
    sf.write(o_p, y_out, SR)
    return y_out, y_ba

if __name__ == "__main__":
    process_audio(sys.argv[1], sys.argv[2], sys.argv[3])
