use rand::random_range;
use std::ops::Range;

pub struct Batches<'a, T> {
    data: &'a [T],
    separators: &'a [T],
    size: Range<usize>,
    index: usize,
}

impl<'a, T: PartialEq> Batches<'a, T> {
    pub fn new(data: &'a [T], separators: &'a [T], size: Range<usize>) -> Self {
        Self {
            data,
            separators,
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
            let start = self.index;

            // ---- Modus 1: Separator ----
            if !self.separators.is_empty() {
                while self.index < len && !self.separators.contains(&self.data[self.index]) {
                    self.index += 1;
                }

                if self.index < len {
                    let end = self.index;
                    self.index += 1; // Separator konsumieren

                    let sentence = &self.data[start..=end]; // inkl. Separator

                    // Nur zurückgeben, wenn Satz > 2 Tokens
                    if sentence.len() > 2 {
                        let input = &sentence[..sentence.len() - 1];
                        let target = &sentence[1..];
                        return Some((input, target));
                    } else {
                        continue; // zu kurz -> nächste Iteration
                    }
                }
            }

            // ---- Modus 2: Random-Länge oder Reststück ----
            let remaining = len - start;

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

            let input_end = start + length - 1;
            let input = &self.data[start..input_end];
            let target = &self.data[start + 1..start + length];

            self.index = start + length;

            return Some((input, target));
        }

        None
    }
}
