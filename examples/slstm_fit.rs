//! Why the fused sLSTM path is or is not available at a given width.
//!
//!   cargo run --release --features cuda --example slstm_fit [H...]
//!
//! Both cooperative kernels decline silently — the caller just takes the per-step
//! loop, which costs ~4x per timestep — so a width that falls off the fused path
//! looks like a slow model rather than a missing kernel. This prints the geometry
//! each one derives, and for the forward the shared-memory arithmetic that decides
//! it: a block stages `[4*units, HP]` of `Wh` in bf16, and the whole grid must be
//! co-resident, so the ENTIRE matrix has to fit the device's aggregate shared
//! memory.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this");
}

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::sys::CUdevice_attribute as DA;
    use neural_networks::gpu::{Gpu, ops};

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    let smem_sm = gpu
        .context
        .attribute(DA::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
        .unwrap_or(-1) as usize;
    println!(
        "{} SMs, {} KB shared/SM ({:.2} MB aggregate), {} KB opt-in per block",
        gpu.sm_count,
        smem_sm / 1024,
        (gpu.sm_count * smem_sm) as f64 / (1024.0 * 1024.0),
        gpu.max_shared_optin / 1024,
    );

    let widths: Vec<usize> = {
        let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
        if a.is_empty() { vec![256, 512, 768, 896, 1024, 1280] } else { a }
    };
    let b = 1;
    println!(
        "\n{:>6} {:>10} {:>8} {:>8} {:>10} {:>8}   {}",
        "H", "Wh bf16", "blocks", "units", "smem KB", "staged", "forward / backward"
    );
    for h in widths {
        let wh_bytes = 8 * h * h;
        let fw = ops::slstm_fused_time_geometry(&gpu, h, b);
        let bw = ops::slstm_fused_time_bwd_geometry(&gpu, h, b);
        let (blocks, want_thr, units, rows, smem) = fw.unwrap_or((0, 0, 0, 0, 0));
        // The geometry knows about shared memory; only the compiled kernel knows how
        // many registers a thread wants, and a block that cannot be placed at all
        // takes the whole cooperative launch down. Ask, rather than report a fused
        // path the launch will quietly decline.
        let threads = fw.and_then(|_| {
            let f = gpu.kernels.specialized(&gpu.context, "slstm_fused_time", &[
                ("SLSTM_H", h),
                ("SLSTM_B", b),
                ("SLSTM_RS", rows),
            ])?;
            f.set_attribute(
                cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem as i32,
            )
            .ok()?;
            ops::fused_fwd_threads_for(&gpu, &f, want_thr, b * units, blocks, smem)
        });
        let staged = match threads {
            Some(_) => format!("{rows}/{}", ops::fused_hp(h)),
            None => "-".into(),
        };
        println!(
            "{h:>6} {:>9.2}M {blocks:>8} {units:>8} {:>9.1} {staged:>8}   {} / {}",
            wh_bytes as f64 / (1024.0 * 1024.0),
            smem as f64 / 1024.0,
            match threads {
                Some(t) => format!("fused @{t}"),
                None => "PER-STEP".into(),
            },
            if bw.is_some() { "fused" } else { "PER-STEP" },
        );
    }
}
