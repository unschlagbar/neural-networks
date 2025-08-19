use std::ops::Range;

pub struct Batches<'a, T> {
    data: &'a [T],
    size: Range<usize>,
    index: usize,
}

impl<'a, T> Batches<'a, T> {
    pub fn new(data: &'a [T], size: Range<usize>) -> Self {
        Self {
            data,
            size,
            index: 0,
        }
    }
}

impl<'a, T> Iterator for Batches<'a, T> {
    type Item = (&'a [T], &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.data.len() - self.index;
        if remaining < 1 {
            return None;
        }
        let max_possible_size = remaining - 1;
        if max_possible_size < self.size.start {
            return None;
        }

        let size: usize;
        if self.size.end == usize::MAX {
            size = max_possible_size;
        } else {
            let choose_end = self.size.end.min(max_possible_size + 1);
            if self.size.start >= choose_end {
                return None;
            }
            size = if self.size.start == self.size.end {
                self.size.start
            } else {
                rand::random_range(self.size.start..choose_end)
            };
        }

        let first = &self.data[self.index..self.index + size];
        let second = &self.data[self.index + 1..self.index + size + 1];

        self.index += size;
        Some((first, second))
    }
}
