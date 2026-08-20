//! What the fused mLSTM kernels cost in shared memory at each head width.
//!
//!   cargo run --release --features cuda --example mlstm_path_check
//!
//! Both head dims are tiled, so the footprint is flat in `dqk` — this is the table
//! that says a given `WORD_HIDDEN` still fits inside what a block can opt into.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this check");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::gpu::{Gpu, ops};

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };
    let limit = gpu.max_shared_optin;
    println!("max_shared_optin = {} B ({:.1} KB)", limit, limit as f64 / 1024.0);
    println!("{:>12} {:>6} {:>14} {:>14} {:>10}", "WORD_HIDDEN", "dqk", "fw_par smem", "bw_par smem", "fits");

    for hidden in [512usize, 768, 1024, 2048, 4096] {
        let heads = 8;
        let dqk = hidden / heads;
        let (fw, bw) = ops::mlstm_fused_smem_parts(ops::FUSED_MAX_L, dqk, dqk, heads);
        let bytes = fw.max(bw);
        println!(
            "{:>12} {:>6} {:>11.1} KB {:>11.1} KB {:>10}",
            hidden,
            dqk,
            fw as f64 / 1024.0,
            bw as f64 / 1024.0,
            if bytes <= limit { "yes" } else { "NO" },
        );
    }
}
