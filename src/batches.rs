use rand::{rng, seq::SliceRandom};


pub struct Batches<'a, T> {
    data: &'a [T],
    seq_len: usize,
    index: usize,
}

impl<'a, T: PartialEq> Batches<'a, T> {
    pub fn new(data: &'a [T], seq_len: usize) -> Self {
        Self {
            data,
            seq_len,
            index: 0,
        }
    }
}

impl<'a, T: PartialEq> Iterator for Batches<'a, T> {
    type Item = (&'a [T], &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.data.len() - self.index;

        if remaining < 3 {
            return None;
        }

        let len = self.seq_len.clamp(3, remaining);

        let input_end = self.index + len;
        let input = &self.data[self.index..input_end - 1];
        let target = &self.data[self.index + 1..input_end];

        self.index += len / 2;

        Some((input, target))
    }
}

pub struct RandomBatches<'a> {
    data: Vec<(&'a [u16], &'a [u16])>,
    index: usize,
}

impl<'a> RandomBatches<'a> {
    pub fn new(seq_len: usize, raw_data: &'a Vec<Vec<u16>>) -> Self {
        let mut data: Vec<(&'a [u16], &'a [u16])> = Vec::new();

        for seq in raw_data {
            let mut index = 0;
            loop {
                let remaining = seq.len() - index;
                if remaining < 3 {
                    break;
                }
                let len = seq_len.clamp(3, remaining);
                let input = &seq[index..index + len - 1];
                let target = &seq[index + 1..index + len];
                data.push((input, target));
                index += len / 2;
            }
        }

        data.shuffle(&mut rng());

        Self {
            data,
            index: 0,
        }
    }
}

impl<'a> Iterator for RandomBatches<'a> {
    type Item = Vec<(&'a [u16], &'a [u16])>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.data.len() {
            let out = self.data[self.index..self.index + 50].to_vec();
            self.index += 50;
            Some(out)
        } else {
            None
        }
    }
}