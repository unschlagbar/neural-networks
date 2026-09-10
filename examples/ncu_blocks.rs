//! One backbone mLSTM block and one backbone sLSTM block at the production chunk
//! shape, forward and backward — a small-footprint target for `ncu`.
//!
//!   ncu --set full -k regex:... target/release/examples/ncu_blocks [reps]
//!
//! A whole model under `ncu` is hopeless: kernel replay saves and restores what a
//! kernel may write, and in a 12 GB process every pass of every profiled kernel pays
//! for that. These two blocks run every kernel a backbone chunk runs, at the same
//! shapes, in a few hundred MB.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    use neural_networks::config::{BACKBONE_CHUNK, WORD_HIDDEN};
    use neural_networks::gpu::arena::TrainingCache;
    use neural_networks::gpu::block::{Block, BlockLike};
    use neural_networks::gpu::hierarchical::up_of;
    use neural_networks::gpu::mlstm::MLstm;
    use neural_networks::gpu::slstm::SLstm;
    use neural_networks::gpu::{GTensor, Gpu, ops, temp};
    use neural_networks::tensor::Tensor;

    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let gpu = Gpu::new().expect("gpu");
    let (t, h, heads) = (BACKBONE_CHUNK, WORD_HIDDEN, 8);
    let dqk = h / heads;
    let l = neural_networks::config::MLSTM_CHUNK.min(ops::FUSED_MAX_L);
    let cache = TrainingCache::new(
        &gpu,
        temp::widest(t, h, heads, dqk, 0),
        temp::widest_small(t, heads),
        temp::widest_chunk(t, t, heads, dqk, h / heads, l),
    );

    let mut blocks: Vec<Box<dyn BlockLike>> = vec![
        Box::new(Block::from_cell(
            &gpu,
            h,
            up_of(h),
            MLstm::new_rand(&gpu, h, h, heads, dqk),
        )),
        Box::new(Block::from_cell(&gpu, h, up_of(h), SLstm::new_rand(&gpu, h, h))),
    ];
    let x = GTensor::from_host(&gpu, &Tensor::random(&[1, t, h], 1.0));
    let dy = GTensor::from_host(&gpu, &Tensor::random(&[1, t, h], 0.1));
    let mut y = GTensor::uninit(&gpu, &[1, t, h]);
    let mut dx = GTensor::uninit(&gpu, &[1, t, h]);
    for _ in 0..reps {
        for blk in blocks.iter_mut() {
            blk.set_carry(true);
            blk.forward(&gpu, &x, &mut y, &cache);
            blk.backward(&gpu, &dy, &mut dx, &cache);
        }
    }
    gpu.stream.synchronize().expect("sync");
    println!("{reps} reps of one mLSTM and one sLSTM block at [1, {t}, {h}]");
}
