"""Make the vendored FlashRNN build against CUDA 13.

`gpu_info.cc` mirrors every `cudaDeviceProp` field into a pybind dict, and CUDA 13
REMOVED several of them, so the `gpu_info2` extension fails to compile -- which takes
the `cuda_fused` backend down with it (the autotuner needs GPU properties). The
`cuda` backend is unaffected.

FlashRNN itself only ever reads four keys (`flashrnn.py`): maxThreadsPerBlock,
regsPerMultiprocessor, multiProcessorCount, sharedMemPerBlockOptin. None are removed
ones, so stubbing the dead fields to 0 changes nothing it depends on.

Idempotent -- safe to re-run. Run after obtaining flashrnn/, which is git-ignored.
"""

import pathlib
import sys

# Fields deleted from cudaDeviceProp in CUDA 13.
REMOVED = [
    "clockRate",
    "deviceOverlap",
    "kernelExecTimeoutEnabled",
    "computeMode",
    "maxTexture1DLinear",
    "memoryClockRate",
    "singleToDoublePrecisionPerfRatio",
    "cooperativeMultiDeviceLaunch",
]

root = pathlib.Path(__file__).resolve().parent / "flashrnn"
src = root / "flashrnn" / "flashrnn" / "gpu_info" / "gpu_info.cc"
if not src.exists():
    sys.exit(f"not found: {src}\nObtain FlashRNN from https://github.com/NX-AI/flashrnn")

text = src.read_text()
changed = 0
for field in REMOVED:
    old = f"prop.{field};"
    new = f"0; // {field}: removed from cudaDeviceProp in CUDA 13"
    if old in text:
        text = text.replace(old, new)
        changed += 1

if changed:
    src.write_text(text)
    print(f"patched {changed} removed cudaDeviceProp field(s) in gpu_info.cc")
else:
    print("gpu_info.cc: already patched (or this CUDA still has the fields)")

# cuda_init.py HARDCODES `arch=compute_80,code=compute_80` and ignores
# TORCH_CUDA_ARCH_LIST, so on an RTX 5080 (sm_120) every kernel ships as sm_80 PTX
# that the driver JITs. Build natively for this card instead.
init = root / "flashrnn" / "flashrnn" / "cuda_init.py"
if init.exists():
    t = init.read_text()
    old = '"arch=compute_80,code=compute_80",'
    new = '"arch=compute_120,code=sm_120",'
    if old in t:
        init.write_text(t.replace(old, new))
        print("cuda_init.py: retargeted compute_80 -> sm_120")
    else:
        print("cuda_init.py: already retargeted")

# `gpu_info.cc` reads `cudaDeviceProp` compiled against CUDA 13 headers, but run.sh links
# torch's 12.8 runtime so the struct is FILLED in 12.8 layout and READ at 13.3 offsets ->
# garbage (regsPerMultiprocessor=1, multiProcessorCount=1, sharedMemPerBlockOptin=2^32+1).
# The fused autotuner then sees 1 SM / 2 registers and correctly finds no valid config,
# which looks exactly like "sm_120 is unsupported" but is purely our header/runtime skew.
#
# torch reads the same properties through its own consistent 12.8 runtime, so overlay the
# four keys FlashRNN actually consumes. Device introspection only -- no kernel is touched.
gi = root / "flashrnn" / "flashrnn" / "gpu_info" / "gpu_info.py"
if gi.exists():
    t = gi.read_text()
    marker = "# --- torch-sourced override (CUDA header/runtime skew) ---"
    if marker not in t:
        t += f'''

{marker}
_flashrnn_orig_get_gpu_info = get_gpu_info


def get_gpu_info(device_id: int) -> dict:
    info = dict(_flashrnn_orig_get_gpu_info(device_id))
    if not info:
        return info
    p = torch.cuda.get_device_properties(device_id)
    info["multiProcessorCount"] = p.multi_processor_count
    info["regsPerMultiprocessor"] = p.regs_per_multiprocessor
    info["sharedMemPerBlockOptin"] = p.shared_memory_per_block_optin
    info["maxThreadsPerBlock"] = p.max_threads_per_multi_processor
    return info
'''
        gi.write_text(t)
        print("gpu_info.py: overlaid 4 device keys from torch")
    else:
        print("gpu_info.py: already overlaid")
