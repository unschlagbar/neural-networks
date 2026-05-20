pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |x, &y| x.max(y));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
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
