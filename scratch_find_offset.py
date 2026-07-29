import librosa
import numpy as np
import scipy.signal
import sys

def find_offset(c_p, b_p):
    y_c, sr = librosa.load(c_p, sr=4000)
    y_b, _ = librosa.load(b_p, sr=4000)
    
    # Try multiple chunks
    for start in [0, 300, 600]:
        c_chunk = y_c[start*4000:(start+60)*4000]
        if len(c_chunk) < 60*4000: break
        corr = scipy.signal.fftconvolve(y_b, c_chunk[::-1], mode='full')
        lags = scipy.signal.correlation_lags(len(y_b), len(c_chunk))
        best_lag = lags[np.argmax(corr)]
        print(f"Start {start}s: Best lag {best_lag} samples -> Offset {best_lag/4000:.1f}s")

if __name__ == "__main__":
    find_offset(sys.argv[1], sys.argv[2])
