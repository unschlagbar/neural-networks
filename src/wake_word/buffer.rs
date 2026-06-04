pub struct RingBuffer {
    data: Vec<f32>,
    head: usize,
    cap: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0.0; capacity],
            head: 0,
            cap: capacity,
        }
    }

    pub fn push(&mut self, samples: &[f32]) {
        for &s in samples {
            self.data[self.head % self.cap] = s;
            self.head += 1;
        }
    }

    /// Returns the last `n` samples in chronological order.
    pub fn last_n(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.cap);
        let start = self.head.saturating_sub(n);
        (0..n)
            .map(|i| self.data[(start + i) % self.cap])
            .collect()
    }

    /// Copy `len` samples starting at absolute position `start`.
    /// Returns `None` if the range hasn't been written yet or has already been overwritten.
    pub fn get_range(&self, start: usize, len: usize) -> Option<Vec<f32>> {
        if self.head < start + len {
            return None; // not written yet
        }
        if start + self.cap < self.head {
            return None; // overwritten
        }
        Some((0..len).map(|i| self.data[(start + i) % self.cap]).collect())
    }

    pub fn total_written(&self) -> usize {
        self.head
    }
}
