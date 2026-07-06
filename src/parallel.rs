// Data-parallel word phases for the hierarchical model.
//
// Encoder words and decoder words are mutually independent (state is reset per
// word), so the words of a window can be split across threads. The layers own
// their recurrent state, scratch buffers and gradient accumulators, which makes
// them unshareable — instead every worker gets a full REPLICA of the stack
// (copied weights, own state, own grad accumulators) and a disjoint slice of
// the shared forward cache, so the phase output lands exactly where the serial
// code would have put it. After a parallel backward phase the per-replica
// gradient accumulators are summed back into the master stack.
//
// Only the raw gradient buffers are reduced; optimizer moments live in the
// master and are untouched. Replicas are rebuilt lazily whenever the master's
// weights change (i.e. after each optimizer step).

use std::range::Range;

use crate::{nn_layer::DynCache, sequential::Sequential};

/// Worker copies of one layer stack, rebuilt lazily after weight updates.
pub struct ReplicaPool {
    pub replicas: Vec<Sequential>,
    dirty: bool,
}

impl ReplicaPool {
    pub fn new() -> Self {
        Self {
            replicas: Vec::new(),
            dirty: true,
        }
    }

    /// Mark the copies stale — call after every weight update on the master.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Bring the replicas up to date with the master's weights if stale.
    /// Existing replicas are refreshed by an in-place weight memcpy; a full
    /// NNFW-round-trip rebuild only happens when the pool size changes.
    pub fn sync(&mut self, master: &Sequential) {
        let workers = 4;
        if self.replicas.len() != workers {
            self.replicas = master.replicas(workers);
        } else if self.dirty {
            for replica in &mut self.replicas {
                replica.copy_weights_from(master);
            }
        }
        self.dirty = false;
    }

    /// Fold every replica's accumulated gradients into the master and clear them.
    pub fn reduce_into(&mut self, master: &mut Sequential) {
        for replica in &mut self.replicas {
            master.add_grads_from(replica);
            replica.clear_grads();
        }
    }
}

/// One worker's share of a parallel word phase: a contiguous run of words plus
/// exactly the cache slots those words occupy (index with `slot - slot_base`).
pub struct WorkerChunk<'a> {
    pub replica: &'a mut Sequential,
    /// Word indices `[start, end)` this chunk owns.
    pub words: Range<usize>,
    pub cache: &'a mut [Vec<Box<dyn DynCache>>],
    pub slot_base: usize,
}

/// Split `n_words` words evenly into one chunk per replica. `slot_ranges[w]`
/// must be cursor-ordered (each word's slots directly follow the previous
/// word's), so every chunk's cache slots form one contiguous run.
pub fn chunk_words<'a>(
    replicas: &'a mut [Sequential],
    slot_ranges: &[Range<usize>],
    n_words: usize,
    mut cache: &'a mut [Vec<Box<dyn DynCache>>],
) -> Vec<WorkerChunk<'a>> {
    let n_chunks = replicas.len().min(n_words);
    let mut chunks = Vec::with_capacity(n_chunks);
    let mut replicas = replicas.iter_mut();
    let mut slot_base = 0;
    let mut w0 = 0;
    for i in 0..n_chunks {
        let w1 = (i + 1) * n_words / n_chunks;
        let slot_end = slot_ranges[w1 - 1].end;
        let (chunk_cache, rest) = cache.split_at_mut(slot_end - slot_base);
        cache = rest;
        chunks.push(WorkerChunk {
            replica: replicas.next().unwrap(),
            words: Range { start: w0, end: w1 },
            cache: chunk_cache,
            slot_base,
        });
        slot_base = slot_end;
        w0 = w1;
    }
    chunks
}

/// Split a per-word buffer into sub-slices matching `chunks`' word ranges, so
/// each worker can write its words' entries without overlap.
pub fn split_by_words<'a, T>(chunks: &[WorkerChunk], mut data: &'a mut [T]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(chunks.len());
    let mut base = 0;
    for chunk in chunks {
        let (head, rest) = data.split_at_mut(chunk.words.end - base);
        out.push(head);
        data = rest;
        base = chunk.words.end;
    }
    out
}
