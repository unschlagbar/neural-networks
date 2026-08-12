//! Where does `slstm_fused_time_geometry` accept, and where does it decline?
//!
//!   cargo run --release --features cuda --example geom_probe
//!
//! The fused time kernel is the fast path; when geometry declines, the sLSTM falls
//! back to the graph (or eager). This prints the accept/decline map over the (H, B)
//! grid we care about, so the batching plan knows in advance which B it can use.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda`");
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

    println!("SMs: {}   max shared opt-in: {} B", gpu.sm_count, gpu.max_shared_optin);
    println!();

    let hs = [256usize, 384, 512, 768, 1024];
    let bs = [1usize, 2, 4, 8, 16, 32];

    print!("{:>6}", "H\\B");
    for b in bs {
        print!("{:>12}", b);
    }
    println!();

    for h in hs {
        print!("{:>6}", h);
        for b in bs {
            match ops::slstm_fused_time_geometry(&gpu, h, b) {
                Some((blocks, _threads, cpb, smem)) => {
                    let _ = cpb;
                    print!("{:>12}", format!("{}bl/{}K", blocks, smem / 1024));
                }
                None => print!("{:>12}", "DECLINE"),
            }
        }
        println!();
    }

    println!();
    println!("backward geometry:");
    print!("{:>6}", "H\\B");
    for b in bs {
        print!("{:>12}", b);
    }
    println!();
    for h in hs {
        print!("{:>6}", h);
        for b in bs {
            match ops::slstm_fused_time_bwd_geometry(&gpu, h, b) {
                Some((blocks, _threads, _cpb, smem, stage_dg)) => {
                    let tag = if stage_dg { "" } else { "*" };
                    print!("{:>12}", format!("{}bl/{}K{}", blocks, smem / 1024, tag));
                }
                None => print!("{:>12}", "DECLINE"),
            }
        }
        println!();
    }
}
