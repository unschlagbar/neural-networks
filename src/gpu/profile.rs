//! Capture-range control for Nsight Systems.
//!
//! `nsys profile --capture-range=cudaProfilerApi` records only what happens
//! between `cuProfilerStart` and `cuProfilerStop`, so a report holds a
//! steady-state training window instead of NVRTC compilation, the first
//! allocations and the pool warmup that dominate the start of a run.
//!
//! Off unless `GPU_NSYS` is set:
//!
//! ```text
//! GPU_NSYS=1 GPU_NSYS_SKIP=10 GPU_NSYS_WINDOWS=1 \
//!   nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
//!                -o step ./target/release/neural-networks <<< hg
//! ```

use crate::gpu::Gpu;

pub struct Capture {
    /// Windows to run before the capture opens.
    skip: usize,
    /// Windows to capture.
    windows: usize,
    seen: usize,
    running: bool,
}

impl Capture {
    /// `None` unless `GPU_NSYS` is set. `GPU_NSYS_SKIP` (default 10) is the
    /// warmup, `GPU_NSYS_WINDOWS` (default 1) the length of the capture.
    pub fn from_env() -> Option<Self> {
        std::env::var("GPU_NSYS").ok()?;
        let num = |k: &str, d: usize| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let c = Capture {
            skip: num("GPU_NSYS_SKIP", 10),
            windows: num("GPU_NSYS_WINDOWS", 1).max(1),
            seen: 0,
            running: false,
        };
        println!(
            "nsys capture: skipping {} window(s), then profiling {} — the process \
             exits when the capture closes.",
            c.skip, c.windows
        );
        if c.windows < crate::config::BATCH_SIZE {
            println!(
                "  (the optimizer step runs every {} windows, so a shorter capture \
                 usually holds none — set GPU_NSYS_WINDOWS={} to include one.)",
                crate::config::BATCH_SIZE,
                crate::config::BATCH_SIZE
            );
        }
        Some(c)
    }

    /// Call at the top of a window. Opens the capture on the first window past
    /// the warmup, after a synchronize so no warmup work lands inside it.
    pub fn before_window(&mut self, gpu: &Gpu) {
        if !self.running && self.seen == self.skip {
            gpu.stream.synchronize().expect("sync before capture");
            unsafe { profiler_start() };
            self.running = true;
            println!("nsys capture started (window {}).", self.seen);
        }
        self.seen += 1;
    }

    /// Call at the end of a window. Closes the capture — and the process, so
    /// nsys sees `--capture-range-end=stop` and finalizes at once — once the
    /// requested number of windows is in it.
    pub fn after_window(&mut self, gpu: &Gpu) {
        if self.running && self.seen >= self.skip + self.windows {
            gpu.stream.synchronize().expect("sync before capture end");
            unsafe { profiler_stop() };
            println!("nsys capture stopped after {} window(s).", self.windows);
            std::process::exit(0);
        }
    }
}

unsafe fn profiler_start() {
    let r = unsafe { cudarc::driver::sys::cuProfilerStart() };
    check(r, "cuProfilerStart");
}

unsafe fn profiler_stop() {
    let r = unsafe { cudarc::driver::sys::cuProfilerStop() };
    check(r, "cuProfilerStop");
}

fn check(r: cudarc::driver::sys::CUresult, what: &str) {
    if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        eprintln!("{what} failed: {r:?}");
    }
}
