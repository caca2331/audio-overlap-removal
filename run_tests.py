"""
Run tests — generates test cases, evaluates all, and logs results to a file.

Usage:
    python run_tests.py
"""

import datetime
import io
import os
import subprocess
import sys


LOG_FILE = "extra-test-out/results.log"


class TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    os.makedirs("extra-test-out", exist_ok=True)

    # --- Step 1: Generate test cases ---
    print("=" * 70)
    print("STEP 1: Generating test cases")
    print("=" * 70)
    result = subprocess.run(
        [sys.executable, "generate_tests.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("ERROR: generate_tests.py failed.")
        sys.exit(1)

    # --- Step 2: Run evaluation, tee output to log ---
    print("\n" + "=" * 70)
    print("STEP 2: Running evaluation")
    print("=" * 70 + "\n")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 70}\n")
        log.write(f"Run: {timestamp}\n")
        log.write(f"{'=' * 70}\n\n")

        old_stdout = sys.stdout
        sys.stdout = TeeWriter(log, old_stdout)

        try:
            # Import and run evaluate inside the tee context
            from evaluate import main as eval_main
            eval_main()
        finally:
            sys.stdout = old_stdout

    print(f"\nResults appended to {LOG_FILE}")


if __name__ == "__main__":
    main()
