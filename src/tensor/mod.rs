//! Clean-room tensor system for the batched NN experiment (`nn2`).
//!
//! Design goals, in order:
//!   1. A single row-major, contiguous `f32` tensor type carrying its own shape.
//!   2. Batched math: the leading dimension is the batch axis, so a per-timestep
//!      matrix-vector product in the old `nn` code becomes a matrix-matrix
//!      product here (arithmetic intensity ~1 -> ~B).
//!   3. No autograd. Layers implement forward/backward by hand, exactly like the
//!      old system, but on `[B, F]` tensors instead of `&[f32]` vectors.
//!
//! This is deliberately kept independent from `iron_oxide::Matrix` and the old
//! `nn` module so the two systems can be benchmarked head to head.

pub mod gemm;

use rand::random_range;

/// Maximum tensor rank the system supports. The NN paths use `[B, F]` (rank 2)
/// and `[B, T, F]` (rank 3); the spare slot leaves room for a head axis
/// (e.g. `[B, T, heads, d]`) without growing the struct again.
pub const MAX_RANK: usize = 4;

/// A dense, row-major, contiguous tensor of `f32`.
///
/// The shape is a fixed-size stack array (no heap allocation for the shape,
/// which matters given how many short-lived tensors the batched paths create);
/// only the first `rank` entries are meaningful, the rest are zero. The NN paths
/// use `[B, F]` (a batch of feature vectors) and `[B, T, F]` (a batch of
/// sequences) almost exclusively.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    /// Dimension sizes; only `shape[..rank]` are valid, trailing slots are 0.
    pub shape: [usize; MAX_RANK],
    pub rank: usize,
    pub data: Vec<f32>,
}

/// Copy `dims` into a zero-filled `[usize; MAX_RANK]`, checking the rank bound.
#[inline]
fn shape_array(dims: &[usize]) -> [usize; MAX_RANK] {
    assert!(
        dims.len() <= MAX_RANK,
        "rank {} exceeds MAX_RANK {MAX_RANK}",
        dims.len()
    );
    let mut s = [0usize; MAX_RANK];
    s[..dims.len()].copy_from_slice(dims);
    s
}

impl Tensor {
    /// Build from an explicit shape and backing data. Panics if they disagree.
    pub fn new(dims: &[usize], data: Vec<f32>) -> Self {
        let n: usize = dims.iter().product();
        assert_eq!(
            n,
            data.len(),
            "Tensor::new — shape {dims:?} implies {n} elements but data has {}",
            data.len()
        );
        Self {
            shape: shape_array(dims),
            rank: dims.len(),
            data,
        }
    }

    /// All-zeros tensor of the given shape.
    pub fn zeros(dims: &[usize]) -> Self {
        let n: usize = dims.iter().product();
        Self {
            shape: shape_array(dims),
            rank: dims.len(),
            data: vec![0.0; n],
        }
    }

    /// Uniform `[-scale, scale)` init — used for weight matrices.
    pub fn random(dims: &[usize], scale: f32) -> Self {
        let n: usize = dims.iter().product();
        let data = (0..n).map(|_| random_range(-scale..scale)).collect();
        Self {
            shape: shape_array(dims),
            rank: dims.len(),
            data,
        }
    }

    /// Xavier/Glorot-uniform init for a `[fan_in, fan_out]` weight matrix.
    /// Matches the old `LinearLayer::new` scale so both systems start comparably.
    pub fn xavier(fan_in: usize, fan_out: usize) -> Self {
        let scale = (6.0 / (fan_in as f32 + fan_out as f32)).sqrt();
        Self::random(&[fan_in, fan_out], scale)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The valid dimension sizes as a slice (`shape[..rank]`).
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Rows of a 2D tensor (panics if not rank 2).
    #[inline]
    pub fn rows(&self) -> usize {
        assert_eq!(self.rank(), 2, "rows(): tensor is rank {}", self.rank());
        self.shape[0]
    }

    /// Columns of a 2D tensor (panics if not rank 2).
    #[inline]
    pub fn cols(&self) -> usize {
        assert_eq!(self.rank(), 2, "cols(): tensor is rank {}", self.rank());
        self.shape[1]
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Reinterpret the buffer under a new shape (same element count). Cheap:
    /// no data is moved, since everything is row-major contiguous.
    pub fn reshape(&self, dims: &[usize]) -> Self {
        let n: usize = dims.iter().product();
        assert_eq!(
            n,
            self.data.len(),
            "reshape {:?} -> {dims:?} changes element count",
            self.dims()
        );
        Self {
            shape: shape_array(dims),
            rank: dims.len(),
            data: self.data.clone(),
        }
    }

    /// Row `r` of a 2D tensor as a slice.
    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        let c = self.cols();
        &self.data[r * c..(r + 1) * c]
    }

    /// Set every element to zero (reused across steps to avoid reallocation).
    pub fn zero_(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Reuse this tensor's buffer under `dims` when the element count is
    /// unchanged (just relabels the shape — no allocation, contents left as-is
    /// for the caller to overwrite); otherwise reallocate as zeros. Used to pool
    /// scratch/state buffers across forward calls so steady-state training does
    /// no per-step allocation.
    pub fn fit(&mut self, dims: &[usize]) {
        let n: usize = dims.iter().product();
        if self.data.len() == n {
            self.shape = shape_array(dims);
            self.rank = dims.len();
        } else {
            *self = Tensor::zeros(dims);
        }
    }
}
