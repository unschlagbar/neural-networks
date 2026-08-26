"""Run ONE (backend, shape) config and print one line. A segfault in the fused
backend kills the process, so each config gets its own so the sweep survives."""
import os, pathlib, sys, time
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "flashrnn"))
import torch
from flashrnn import flashrnn

backend = sys.argv[1]
B, T, N, D = (int(x) for x in sys.argv[2:6])
dev = torch.device("cuda"); dt = torch.bfloat16
G, S = 4, 4
Wx = torch.randn([B, T, G, N, D], device=dev, dtype=dt, requires_grad=True)
R  = (torch.randn([G, N, D, D], device=dev, dtype=dt) * (D ** -0.5)).requires_grad_()
b  = torch.randn([G, N, D], device=dev, dtype=dt, requires_grad=True)
s0 = torch.zeros([S, B, 1, N, D], device=dev, dtype=dt, requires_grad=True)

def once(bwd):
    st, _ = flashrnn(Wx, R, b, states=s0, function="slstm", backend=backend)
    if bwd:
        st[0].sum().backward()
        Wx.grad = None; R.grad = None; b.grad = None

for _ in range(5):
    once(True)
torch.cuda.synchronize()
out = {}
for label, bwd in (("fwd", False), ("fwd+bwd", True)):
    best = float("inf")
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            once(bwd)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3 / 20)
    out[label] = best
print(f"RESULT {backend} B={B} T={T} N={N} D={D} fwd={out['fwd']:.3f} fwd+bwd={out['fwd+bwd']:.3f}")
