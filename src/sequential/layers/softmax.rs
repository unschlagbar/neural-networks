pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

/// Cross-entropy of `target` under softmax(logits), without allocating:
/// `logsumexp(logits) - logits[target]` (= `-ln softmax(logits)[target]`).
pub fn nll_from_logits(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for &v in logits {
        sum += (v - max).exp();
    }
    (max + sum.ln()) - logits[target]
}

/// Nucleus (top-p) sampling: temperature-scale the logits, softmax them, then
/// sample from the smallest set of tokens whose cumulative probability reaches
/// `top_p` (renormalized implicitly by drawing from `0..cum`).
pub fn sample_top_p(logits: &[f32], temperature: f32, top_p: f32) -> usize {
    let scaled: Vec<f32> = logits.iter().map(|&v| v / temperature.max(1e-8)).collect();
    let q = softmax(&scaled);

    let mut idx: Vec<usize> = (0..q.len()).collect();
    idx.sort_unstable_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap());

    let mut cum = 0.0;
    let candidates: Vec<usize> = idx
        .iter()
        .copied()
        .take_while(|&i| {
            if cum >= top_p {
                false
            } else {
                cum += q[i];
                true
            }
        })
        .collect();

    let r = rand::random_range(0.0..cum);
    let mut acc = 0.0;
    for &i in &candidates {
        acc += q[i];
        if acc >= r {
            return i;
        }
    }
    candidates[0]
}

/// Returns a heap-allocated softmax result — used for sampling temperature scaling.
pub fn softmax(vec: &[f32]) -> Box<[f32]> {
    let max = vec.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y));
    let mut sum = 0.0;
    let mut out: Box<[f32]> = vec
        .iter()
        .map(|&x| {
            let e = (x - max).exp();
            sum += e;
            e
        })
        .collect();
    out.iter_mut().for_each(|x| *x /= sum);
    out
}
