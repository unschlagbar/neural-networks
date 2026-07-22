//! What do the fused mLSTM kernels actually cost the machine?
//!
//!   cargo run --release --features cuda --example gpu_occupancy
//!
//! Registers per thread, shared memory per block, register SPILLS, and how many
//! blocks the driver will co-resident on one SM — all read from the driver
//! (`cuFuncGetAttribute` / `cuOccupancyMaxActiveBlocksPerMultiprocessor`), not
//! guessed from the source. `waves` is grid / (blocks-per-SM * SMs): under 1.0 the
//! GPU is not even filled once.
//!
//! The three limits to read it against, printed at the top: threads/SM, shared
//! memory/SM, and registers/SM. Whichever the kernel hits first is what caps its
//! occupancy — that is the number to attack.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this");
}

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::sys::CUdevice_attribute as DA;
    use cudarc::driver::sys::CUfunction_attribute as FA;
    use neural_networks::gpu::{Gpu, ops};

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    let dev = |a: DA| gpu.context.attribute(a).unwrap_or(-1);
    let sms = dev(DA::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    let thr_sm = dev(DA::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    let smem_sm = dev(DA::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    let regs_sm = dev(DA::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    let smem_blk_opt = dev(DA::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    let (cc_maj, cc_min) = (
        dev(DA::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        dev(DA::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
    );

    println!("device: sm_{cc_maj}{cc_min}, {sms} SMs");
    println!(
        "per SM: {thr_sm} threads ({} warps), {} KB shared, {regs_sm} registers",
        thr_sm / 32,
        smem_sm / 1024,
    );
    println!(
        "per block: {} KB shared max (opt-in)\n",
        smem_blk_opt / 1024
    );

    // The backbone's real shape: one window, WORD_HIDDEN wide, 8 heads.
    let (b, t, heads, dqk) = (1usize, 2047usize, 8usize, 64usize);
    let d = neural_networks::config::WORD_HIDDEN;
    let (bh, dhv) = (b * heads, d / heads);
    let l = ops::FUSED_MAX_L;

    println!("backbone shape: BH={bh} T={t} dqk={dqk} dhv={dhv} L={l}\n");
    println!(
        "{:<20} {:>4} {:>5} {:>7} {:>7} {:>7} {:>8} {:>7} {:>6} {:>6}",
        "kernel", "thr", "regs", "spill", "smem KB", "grid", "blocks/SM", "warps", "occ%", "waves",
    );

    for (name, f, smem, grid) in ops::mlstm_fused_kernels(&gpu, l, dqk, dhv, bh, t) {
        let threads = ops::fused_threads(name);
        let fa = |a: FA| f.get_attribute(a).unwrap_or(-1);
        let regs = fa(FA::CU_FUNC_ATTRIBUTE_NUM_REGS);
        // Local memory per thread: nonzero means the kernel SPILLED registers to
        // (cached, but off-chip) local memory. That is the one number here that is
        // never acceptable in an inner-loop kernel.
        let spill = fa(FA::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES);

        let blocks_sm = f
            .occupancy_max_active_blocks_per_multiprocessor(threads, smem as usize, None)
            .map(|v| v as i32)
            .unwrap_or(-1);

        let nblocks = (grid.0 * grid.1) as i32;
        let warps = blocks_sm * (threads as i32) / 32;
        let occ = 100.0 * warps as f64 / (thr_sm / 32) as f64;
        let waves = nblocks as f64 / (blocks_sm.max(1) * sms) as f64;

        println!(
            "{name:<20} {threads:>4} {regs:>5} {spill:>7} {:>7.1} {:>7} {blocks_sm:>8} {warps:>7} {occ:>5.0}% {waves:>6.2}",
            smem as f64 / 1024.0,
            nblocks,
        );
    }

    println!(
        "\nspill != 0 => registers went to local memory (fix first).\n\
         occ%  => achieved warps/SM vs the hardware max; low occ is fine ONLY if the\n\
                  kernel is compute-dense enough to hide latency without it.\n\
         waves => grid / (blocks-per-SM x SMs); < 1.0 means the GPU is never filled."
    );
}
