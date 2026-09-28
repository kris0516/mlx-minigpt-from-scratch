"""Step 01: verify that Python is running natively on Apple Silicon."""

import platform
import sys


print(f"System : {platform.system()}")
print(f"Machine: {platform.machine()}")
print(f"Python : {sys.version.split()[0]}")


if platform.system() != "Darwin":
    raise RuntimeError("This project is designed for macOS.")

if platform.machine() != "arm64":
    raise RuntimeError(
        "Python is not running as native arm64. "
        "Check whether VS Code is using an x86_64/Rosetta interpreter."
    )

print("OK     : native Apple Silicon Python detected.")
