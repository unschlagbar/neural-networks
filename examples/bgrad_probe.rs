//! Does cuBLASLt's `BGRADA` epilogue accept the weight-gradient GEMM we actually run,
//! and is it free?
//!
//! `dW += Xᵀ·dY` needs `db += Σ_rows dY` alongside it, and today that is a separate
//! full pass over `dY` (`add_col_sum`, ~10 % of GPU time). The fused answer — PyTorch's
//! `gemm_and_bias` path, and TransformerEngine's wgrad, which is where the pattern
//! comes from — asks cuBLASLt for the column sum in the GEMM's epilogue, where the
//! operand is already in registers.
//!
//! Two things have to hold for that to pay here, and they are separate questions:
//!
//! 1. the epilogue must accept *our* operand layout — with this project's column-major
//!    operand swap `dY` is `A` in the non-transposed form (`NT`), not the `TN` the
//!    fused epilogues are usually documented against; and
//! 2. cuBLASLt's chosen algorithm must be no slower than the `cublasGemmEx` the legacy
//!    path gets. Moving the GEMM to Lt changes which kernel runs, so the epilogue can
//!    be free and the change still lose.
//!
//! The `Lt no epilogue` column separates the two: against `gemm` it is the cost of the
//! backend switch, against `Lt + BGRADA` the cost of the epilogue.
//!
//!   cargo run --release --features cuda --example bgrad_probe

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("build with `--features cuda` to run this probe");
}

#[cfg(feature = "cuda")]
fn main() {
    use std::time::Instant;

    use cudarc::cublaslt::{MatmulShared, result, sys};
    use cudarc::driver::DevicePtr;
    use neural_networks::gpu::{GTensor, Gpu, ops};
    use neural_networks::tensor::Tensor;

    let gpu = match Gpu::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("no GPU: {e}");
            return;
        }
    };

    const IT: usize = 300;
    // Weight-gradient shapes from one real step: (rows, in, out).
    let shapes = [
        (512usize, 768usize, 768usize),
        (512, 768, 2048),
        (512, 768, 16),
        (2045, 256, 256),
        (2045, 256, 682),
    ];

    println!(
        "{:>16} {:>10} {:>13} {:>15} {:>13}",
        "rows x in x out", "gemm", "gemm+colsum", "Lt no epilogue", "Lt + BGRADA"
    );

    for (rows, input, output) in shapes {
        let x_h = Tensor::random(&[rows, input], 0.5);
        let dy_h = Tensor::random(&[rows, output], 0.5);

        let x = GTensor::from_host(&gpu, &x_h);
        let dy = GTensor::from_host(&gpu, &dy_h);
        let mut x_b = GTensor::uninit(&gpu, &[rows, input]);
        x_b.store(&gpu, &x);
        let mut dy_b = GTensor::uninit(&gpu, &[rows, output]);
        dy_b.store(&gpu, &dy);

        let dw = GTensor::zeros(&gpu, &[input, output]);
        let mut db = GTensor::zeros(&gpu, &[output]);
        let ws = GTensor::zeros(&gpu, &[1024 * 1024]); // cudarc keeps its own private
        let ws_size = ws.capacity() * 4;

        // Scoped so the `SyncOnDrop` guards release the borrows; the allocations
        // outlive them, so the raw pointers stay valid.
        let ptr = |t: &GTensor<f32>| {
            let (p, _g) = t.buf.device_ptr(&gpu.stream);
            p
        };
        let bptr = |t: &GTensor<u16>| {
            let (p, _g) = t.buf.device_ptr(&gpu.stream);
            p
        };
        let (pdb, pc, pw) = (ptr(&db), ptr(&dw), ptr(&ws));
        let (pa, pb) = (bptr(&dy_b), bptr(&x_b));

        // cuBLASLt is column-major and we want D = dWᵀ (`[out, in]` col-major is
        // `dW` `[in, out]` row-major), so A is `dY` and B is `X`:
        //   D(m=out, n=in) = A(out, N) · B(N, in),  k = N = rows
        // `dY` row-major `[N, out]` reads as col-major `out x N`, ld = out, op N.
        // `X`  row-major `[N, in]`  reads as col-major `in x N`,  ld = in,  op T.
        let (m, n, k) = (output as u64, input as u64, rows as u64);
        let a_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_16BF, m, k, output as i64)
                .expect("a layout");
        let b_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_16BF, n, k, input as i64)
                .expect("b layout");
        let c_layout =
            result::create_matrix_layout(sys::cudaDataType_t::CUDA_R_32F, m, n, output as i64)
                .expect("c layout");

        let pref = result::create_matmul_pref().expect("pref");
        unsafe {
            result::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws_size) as *const _ as *const _,
                std::mem::size_of::<usize>(),
            )
            .expect("workspace size");
        }

        // Time one descriptor; `bgrad = false` is the plain GEMM, so the two differ in
        // nothing but the epilogue.
        let timed = |bgrad: bool| -> Option<f64> {
            let desc = result::create_matmul_desc(
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cudaDataType_t::CUDA_R_32F,
            )
            .expect("desc");
            let (op_n, op_t) = (0i32, 1i32);
            let set = |attr, p: *const std::ffi::c_void, sz| unsafe {
                result::set_matmul_desc_attribute(desc, attr, p, sz).expect("desc attr");
            };
            use sys::cublasLtMatmulDescAttributes_t as A;
            set(
                A::CUBLASLT_MATMUL_DESC_TRANSA,
                (&op_n) as *const _ as *const _,
                4,
            );
            set(
                A::CUBLASLT_MATMUL_DESC_TRANSB,
                (&op_t) as *const _ as *const _,
                4,
            );
            if bgrad {
                // Reduce A over k into a vector of D's row count (= out) — exactly db.
                let epi = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADA;
                let bias_ty = sys::cudaDataType_t::CUDA_R_32F;
                set(
                    A::CUBLASLT_MATMUL_DESC_EPILOGUE,
                    (&epi) as *const _ as *const _,
                    std::mem::size_of::<sys::cublasLtEpilogue_t>(),
                );
                set(
                    A::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    (&bias_ty) as *const _ as *const _,
                    std::mem::size_of::<sys::cudaDataType_t>(),
                );
                set(
                    A::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&pdb) as *const _ as *const _,
                    std::mem::size_of::<u64>(),
                );
            }

            let heuristic = unsafe {
                result::get_matmul_algo_heuristic(
                    *gpu.blas_lt.handle(),
                    desc,
                    a_layout,
                    b_layout,
                    c_layout,
                    c_layout,
                    pref,
                )
            }
            .ok()?;

            // beta = 1: `dw` is an accumulator, as in the legacy path.
            let (alpha, beta) = (1.0f32, 1.0f32);
            let run = || unsafe {
                result::matmul(
                    *gpu.blas_lt.handle(),
                    desc,
                    (&alpha) as *const _ as *const _,
                    (&beta) as *const _ as *const _,
                    pa as *const _,
                    a_layout,
                    pb as *const _,
                    b_layout,
                    pc as *const _,
                    c_layout,
                    pc as *mut _,
                    c_layout,
                    (&heuristic.algo) as *const _,
                    pw as *mut _,
                    ws_size,
                    gpu.stream.cu_stream() as *mut _,
                )
            };
            run().ok()?;
            for _ in 0..30 {
                run().unwrap();
            }
            gpu.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..IT {
                run().unwrap();
            }
            gpu.stream.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / IT as f64;
            unsafe { result::destroy_matmul_desc(desc).ok() };
            Some(us)
        };

        let plain = timed(false);
        let fused = timed(true);

        // The epilogue's db, against a host column sum. It is reduced from the bf16
        // `A`, so it carries that rounding — the same operand the GEMM contracts.
        let got = db.to_host(&gpu).data;
        let mut want = vec![0.0; output];
        for r in 0..rows {
            for c in 0..output {
                want[c] += dy_h.data[r * output + c];
            }
        }
        let err = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w).abs() / w.abs().max(1.0))
            .fold(0.0f32, f32::max);

        // The two kernels in use today, alone and together.
        let mut dw_ref = GTensor::zeros(&gpu, &[input, output]);
        let mut bench = |with_colsum: bool| {
            let mut go = || {
                ops::matmul_bf16_into(&gpu, ops::MmForm::Tn, &x_b, &dy_b, &mut dw_ref, 1.0);
                if with_colsum {
                    ops::add_col_sum(&gpu, &mut db, &dy);
                }
            };
            for _ in 0..30 {
                go();
            }
            gpu.stream.synchronize().unwrap();
            let t0 = Instant::now();
            for _ in 0..IT {
                go();
            }
            gpu.stream.synchronize().unwrap();
            t0.elapsed().as_secs_f64() * 1e6 / IT as f64
        };
        let gemm = bench(false);
        let split = bench(true);

        let show = |r: Option<f64>| match r {
            Some(us) => format!("{us:.2} us"),
            None => "no algo".to_string(),
        };
        println!(
            "{:>16} {gemm:>7.2} us {split:>10.2} us {:>15} {:>13}   db rel-err {err:.1e}",
            format!("{rows}x{input}x{output}"),
            show(plain),
            show(fused),
        );

        unsafe {
            result::destroy_matmul_pref(pref).ok();
            result::destroy_matrix_layout(a_layout).ok();
            result::destroy_matrix_layout(b_layout).ok();
            result::destroy_matrix_layout(c_layout).ok();
        }
    }
}
