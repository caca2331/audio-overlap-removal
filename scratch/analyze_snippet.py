import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

AUDIO_PATH = "extra-test-out/out_C15S.flac"
OUT_IMG = "extra-test-out/spectrogram_T15S_snippet.png"

# Load snippet 1:30 - 2:00 (90s - 120s)
y, sr = librosa.load(AUDIO_PATH, sr=44100, offset=90, duration=30)

# Compute STFT
D = librosa.stft(y, n_fft=2048, hop_length=512)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot
plt.figure(figsize=(12, 6))
librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of T15S (1:30-2:00)')
plt.tight_layout()
plt.savefig(OUT_IMG)
print(f"Spectrogram saved to {OUT_IMG}")
