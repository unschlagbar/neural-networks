use rand::random_range;
use std::ops::Range;

pub struct Batches<'a, T> {
    data: &'a [T],
    size: Range<usize>,
    index: usize,
}

impl<'a, T: PartialEq> Batches<'a, T> {
    pub fn new(data: &'a [T], size: Range<usize>) -> Self {
        Self {
            data,
            size,
            index: 0,
        }
    }
}

impl<'a, T: PartialEq> Iterator for Batches<'a, T> {
    type Item = (&'a [T], &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.data.len();

        while self.index < len {

            // ---- Modus 2: Random-Länge oder Reststück ----
            let remaining = len - self.index;

            // Wenn zu kurz -> überspringen
            if remaining <= 2 {
                self.index = len; // fertig
                return None;
            }

            let min_len = self.size.start.max(3); // mindestens 3
            let max_len = self.size.end.min(remaining);

            let length = if remaining <= min_len {
                remaining // Reststück, >2 hier gesichert
            } else if min_len >= max_len {
                min_len
            } else {
                random_range(min_len..=max_len)
            };

            let input_end = self.index + length;
            let input = &self.data[self.index..input_end - 1];
            let target = &self.data[self.index + 1..input_end];

            self.index += length;

            return Some((input, target));
        }

        None
    }
}
