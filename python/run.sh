#!/bin/sh
# Run FlashRNN against a CUDA runtime that MATCHES torch's.
#
# The system toolkit is 13.3 but torch ships cu128, so a default build links
# libcudart.so.13 into a process already holding torch's libcudart.so.12. Two runtimes
# means a cudaEvent created by one is a garbage handle to the other, and the first
# cudaStreamWaitEvent segfaults inside libcuda -- with ZERO CUDA errors reported by
# compute-sanitizer, because the fault is host-side.
#
# The shim compiles with the only nvcc present (13.3) but links torch's 12.8 runtime,
# so one runtime owns every stream and event. `bin` must be the whole directory, not a
# symlinked nvcc: nvcc locates cicc/cudafe++ relative to itself and exits 127 otherwise.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SP="$HERE/.venv/lib/python3.14/site-packages/nvidia"
SHIM="$HOME/.cuda128_shim"

if [ ! -e "$SHIM/lib64/libcudart.so" ]; then
    rm -rf "$SHIM"; mkdir -p "$SHIM/lib64"
    ln -sf /opt/cuda/bin "$SHIM/bin"
    ln -sf /opt/cuda/include "$SHIM/include"
    ln -sf "$SP/cuda_runtime/lib/libcudart.so.12" "$SHIM/lib64/libcudart.so"
    ln -sf "$SP/cublas/lib/libcublas.so.12"       "$SHIM/lib64/libcublas.so"
    ln -sf "$SP/cublas/lib/libcublasLt.so.12"     "$SHIM/lib64/libcublasLt.so"
    echo "built CUDA shim at $SHIM" >&2
fi

CUDA_HOME="$SHIM" CUDA_PATH="$SHIM" CUDA_LIB="$SP/cublas/lib" \
    exec "$HERE/.venv/bin/python" -u "$@"
