use rand::{rng, seq::SliceRandom};

// ── Batches — sequential with configurable stride ────────────────────────────
//
//  stride < seq_len  → overlapping windows (more training examples per file)
//  stride == seq_len → non-overlapping (classic behaviour)
//
//  Example with seq_len=100, stride=50:
//    Window 0: [0..99]   → input [0..98],  target [1..99]
//    Window 1: [50..149] → ...
//    Window 2: [100..199]

pub struct Batches<'a, T> {
    data: &'a [T],
    seq_len: usize,
    stride: usize,
    index: usize,
    // ── debug counters ────────────────────────────────────────────────────────
    batch_count: usize,
    total_tokens: usize,
}

impl<'a, T> Batches<'a, T> {
    /// stride == seq_len → no overlap (classic behaviour).
    pub fn new(data: &'a [T], seq_len: usize) -> Self {
        Self::with_stride(data, seq_len, seq_len)
    }

    /// stride < seq_len → overlapping windows for more training examples.
    pub fn with_stride(data: &'a [T], seq_len: usize, stride: usize) -> Self {
        assert!(stride >= 1, "stride must be >= 1");
        Self {
            data,
            seq_len,
            stride,
            index: 0,
            batch_count: 0,
            total_tokens: 0,
        }
    }

    /// Number of windows yielded so far.
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }
    /// Total tokens seen so far (input side only, i.e. window_len - 1 per window).
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Estimate the total number of windows that will be produced.
    /// (Upper bound; last window may be shorter.)
    pub fn estimated_total(&self) -> usize {
        if self.data.len() < 2 {
            return 0;
        }
        (self.data.len().saturating_sub(1) + self.stride - 1) / self.stride
    }
}

impl<'a, T> Iterator for Batches<'a, T> {
    type Item = (&'a [T], &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.data.len().saturating_sub(self.index);

        // At least 2 tokens needed (1 input + 1 target).
        if remaining < 2 {
            return None;
        }

        let window = self.seq_len.clamp(2, remaining);
        let input = &self.data[self.index..self.index + window - 1];
        let target = &self.data[self.index + 1..self.index + window];

        // Stride is capped to the current window to prevent gaps.
        self.index += self.stride.min(window);

        self.batch_count += 1;
        self.total_tokens += input.len();

        Some((input, target))
    }
}

// ── BatchDebugger ─────────────────────────────────────────────────────────────
//
// Wrap any (input, target) batch iterator to get per-epoch statistics.
//
//   let batches = Batches::new(&data, SEQ_LEN);
//   let mut dbg = BatchDebugger::new(batches, "Epoch 1");
//   model.train(&mut dbg, ...);
//   dbg.print_summary();

pub struct BatchDebugger<I> {
    inner: I,
    label: String,
    count: usize,
    min_len: usize,
    max_len: usize,
    total_tokens: usize,
}

impl<'a, I: Iterator<Item = (&'a [u16], &'a [u16])>> BatchDebugger<I> {
    pub fn new(inner: I, label: impl Into<String>) -> Self {
        Self {
            inner,
            label: label.into(),
            count: 0,
            min_len: usize::MAX,
            max_len: 0,
            total_tokens: 0,
        }
    }

    pub fn print_summary(&self) {
        if self.count == 0 {
            println!("[{}] no batches yielded", self.label);
            return;
        }
        println!(
            "[{}] batches={} | tokens={} | seq len min={} avg={} max={}",
            self.label,
            self.count,
            self.total_tokens,
            self.min_len,
            self.total_tokens / self.count,
            self.max_len,
        );
    }
}

impl<'a, I: Iterator<Item = (&'a [u16], &'a [u16])>> Iterator for BatchDebugger<I> {
    type Item = (&'a [u16], &'a [u16]);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        self.count += 1;
        let len = item.0.len();
        self.total_tokens += len;
        if len < self.min_len {
            self.min_len = len;
        }
        if len > self.max_len {
            self.max_len = len;
        }
        Some(item)
    }
}

// ── RandomBatches — shuffled mini-batches from multiple sequences ─────────────

pub struct RandomBatches<'a> {
    data: Vec<(&'a [u16], &'a [u16])>,
    index: usize,
    batch_size: usize,
}

impl<'a> RandomBatches<'a> {
    pub fn new(seq_len: usize, batch_size: usize, raw_data: &'a [Vec<u16>]) -> Self {
        let stride = (seq_len / 2).max(1);
        let mut data: Vec<(&'a [u16], &'a [u16])> = Vec::new();

        for seq in raw_data {
            let mut index = 0;
            loop {
                let remaining = seq.len().saturating_sub(index);
                if remaining < 2 {
                    break;
                }
                let len = seq_len.clamp(2, remaining);
                data.push((&seq[index..index + len - 1], &seq[index + 1..index + len]));
                index += stride;
            }
        }

        data.shuffle(&mut rng());
        Self {
            data,
            index: 0,
            batch_size,
        }
    }

    pub fn total_windows(&self) -> usize {
        self.data.len()
    }
}

impl<'a> Iterator for RandomBatches<'a> {
    type Item = Vec<(&'a [u16], &'a [u16])>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.data.len() {
            return None;
        }
        let end = (self.index + self.batch_size).min(self.data.len());
        let batch = self.data[self.index..end].to_vec();
        self.index = end;
        Some(batch)
    }
}

// ── WordBoundaryBatches — split only at word boundaries ───────────────────────

pub struct WordBoundaryBatches<'a> {
    data: &'a [u16],
    boundary_ids: &'a [u16],
    seq_len: usize,
    index: usize,
}

impl<'a> WordBoundaryBatches<'a> {
    pub fn new(data: &'a [u16], boundary_ids: &'a [u16], seq_len: usize) -> Self {
        Self {
            data,
            boundary_ids,
            seq_len,
            index: 0,
        }
    }
}

impl<'a> Iterator for WordBoundaryBatches<'a> {
    type Item = (&'a [u16], &'a [u16]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index + 1 >= self.data.len() {
            return None;
        }

        let mut end = (self.index + self.seq_len).min(self.data.len());

        // Extend to the next word boundary (up to data end).
        while end < self.data.len() && !self.boundary_ids.contains(&self.data[end - 1]) {
            end += 1;
        }

        let input = &self.data[self.index..end - 1];
        let target = &self.data[self.index + 1..end];

        self.index = end;
        Some((input, target))
    }
}
