#!/bin/bash
# Expose env-speech's pip-installed CUDA libraries without changing shell config.
set -euo pipefail
speech_site=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
speech_cuda_libs=$(python -c 'from pathlib import Path; import sysconfig; print(":".join(str(p) for p in sorted(Path(sysconfig.get_path("purelib")).glob("nvidia/*/lib"))))')
export LD_LIBRARY_PATH="$speech_cuda_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$speech_site/nvidia/cuda_nvcc/bin:$PATH"
exec "$@"
