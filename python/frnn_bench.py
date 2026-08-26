"""Head-to-head: FlashRNN's sLSTM recurrent loop vs ours, at OUR backbone shape.

flashrnn() takes Wx already computed, so this times the recurrent part only --
the same span as our slstm_fused_time / _bwd pair. Backward is included because
that is where most of our step time sits.

N=1, D=H makes their block-diagonal R dense, which is exactly our whr [H, 4H].
"""
import os, pathlib, sys, time
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "flashrnn"))

import torch
from flashrnn import flashrnn

dev = torch.device("cuda")
print("torch", torch.__version__, "| cap", torch.cuda.get_device_capability(0), "|", torch.cuda.get_device_name(0))

G, S = 4, 4  # sLSTM: 4 gates, 4 states (y, c, n, m)

def bench(B, T, N, D, backend, dtype=torch.bfloat16, iters=20, warm=5):
    Wx = torch.randn([B, T, G, N, D], device=dev, dtype=dtype, requires_grad=True)
    R  = torch.randn([G, N, D, D], device=dev, dtype=dtype, requires_grad=True) * (D ** -0.5)
    b  = torch.randn([G, N, D], device=dev, dtype=dtype, requires_grad=True)
    s0 = torch.zeros([S, B, 1, N, D], device=dev, dtype=dtype, requires_grad=True)

    def once(bwd):
        states, _ = flashrnn(Wx, R, b, states=s0, function="slstm", backend=backend)
        if bwd:
            states[0].sum().backward(retain_graph=False)
            Wx.grad = None; R.grad = None; b.grad = None
        return states

    for _ in range(warm):
        once(True)
    torch.cuda.synchronize()

    out = {}
    for label, bwd in (("fwd", False), ("fwd+bwd", True)):
        # min over rounds: SM clock swings hard on this card
        best = float("inf")
        for _ in range(5):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(iters):
                once(bwd)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) * 1e3 / iters)
        out[label] = best
    return out

# (B, T, N, D) -- our backbone chunk is B=1, T=512, H=1024.
shapes = [
    ("backbone B=1 T=512 H=1024 (N=1,D=1024)", 1, 512, 1, 1024),
    ("backbone B=1 T=512 H=1024 (N=4,D=256)",  1, 512, 4, 256),
    ("backbone B=1 T=512 H=1024 (N=16,D=64)",  1, 512, 16, 64),
    ("encoder  B=227 T=8 H=256  (N=1,D=256)",  227, 8, 1, 256),
]

print(f"\n{'shape':<42} {'backend':<12} {'fwd ms':>9} {'fwd+bwd ms':>12}")
for label, B, T, N, D in shapes:
    for backend in ("cuda_fused", "cuda"):
        try:
            r = bench(B, T, N, D, backend)
            print(f"{label:<42} {backend:<12} {r['fwd']:>9.3f} {r['fwd+bwd']:>12.3f}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:90]
            print(f"{label:<42} {backend:<12} {'FAILED':>9}  {msg}")
