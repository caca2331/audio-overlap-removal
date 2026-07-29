import os
import shutil
import subprocess

OUT_DIR = "extra-test-out"
os.makedirs(OUT_DIR, exist_ok=True)

# Source files
SRC_MIYAKO = "extra-asset/miyako.webm"
SRC_YA = "extra-asset/ya.mp3"
SRC_SR = "extra-asset/sr.webm"

BASE_A = f"{OUT_DIR}/base_a.flac"
BASE_B = f"{OUT_DIR}/base_b.flac"

def run_cmd(cmd):
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def mix(a, b, out_c, out_b, filter_complex="[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
        extra_args=None):
    print(f"  Generating {out_c}...")
    cmd = (["ffmpeg", "-y", "-i", a, "-i", b, "-filter_complex", filter_complex,
            "-map", "[a]", "-c:a", "flac"] + (extra_args or []) + [out_c])
    run_cmd(cmd)
    shutil.copy(b, out_b)

# ---------------------------------------------------------------------------
# Base audio extraction
# ---------------------------------------------------------------------------
print("Extracting base audio...")
if not os.path.exists(BASE_A):
    run_cmd(["ffmpeg", "-y", "-ss", "00:10:00", "-i", SRC_MIYAKO,
             "-t", "120", "-c:a", "flac", "-ar", "44100", "-ac", "1", BASE_A])
if not os.path.exists(BASE_B):
    run_cmd(["ffmpeg", "-y", "-ss", "00:01:00", "-i", SRC_YA,
             "-t", "120", "-c:a", "flac", "-ar", "44100", "-ac", "1", BASE_B])

# ---------------------------------------------------------------------------
# T15S (Short) - RUN THIS FIRST
# ---------------------------------------------------------------------------
print("\nT15S: 15-Min Real World (Short)")
run_cmd(["ffmpeg", "-y", "-ss", "300", "-i", SRC_MIYAKO, "-t", "900",
         "-c:a", "flac", "-sample_fmt", "s16", "-ar", "44100", "-ac", "1",
         f"{OUT_DIR}/C15S.flac"])
# Extract entire background to handle any offsets
run_cmd(["ffmpeg", "-y", "-i", SRC_SR,
         "-c:a", "flac", "-sample_fmt", "s16", "-ar", "44100", "-ac", "1",
         f"{OUT_DIR}/B15S.flac"])

# ---------------------------------------------------------------------------
# Synthetic Tests
# ---------------------------------------------------------------------------
TESTS = {
    1:  ("Standard Mix", {}),
    14: ("Network Stutter", {
        "filter_complex": "[1:a]atrim=0:60,asetpts=PTS-STARTPTS[p1];"
                          "[1:a]atrim=59:60,asetpts=PTS-STARTPTS[p2];"
                          "[1:a]atrim=59:60,asetpts=PTS-STARTPTS[p3];"
                          "[1:a]atrim=59:60,asetpts=PTS-STARTPTS[p4];"
                          "[1:a]atrim=60:117,asetpts=PTS-STARTPTS[p5];"
                          "[p1][p2][p3][p4][p5]concat=n=5:v=0:a=1[b_mod];"
                          "[0:a][b_mod]amix=inputs=2:duration=longest:normalize=0[a]"}),
}

for i in [1, 14]:
    desc, kwargs = TESTS[i]
    print(f"T{i}: {desc}")
    mix(BASE_A, BASE_B, f"{OUT_DIR}/C{i}.flac", f"{OUT_DIR}/B{i}.flac", **kwargs)

# ---------------------------------------------------------------------------
# T15: 90-Min Real World (Blind)
# ---------------------------------------------------------------------------
print("\nT15: 90-Min Real World (Blind)")
run_cmd(["ffmpeg", "-y", "-i", SRC_MIYAKO, "-t", "5400",
         "-c:a", "flac", "-sample_fmt", "s16", "-ar", "44100", "-ac", "1",
         f"{OUT_DIR}/C15.flac"])
# Use same B15S as background since it's the full OST
shutil.copy(f"{OUT_DIR}/B15S.flac", f"{OUT_DIR}/B15.flac")

print("\nTests generated.")
