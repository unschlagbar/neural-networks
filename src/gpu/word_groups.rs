//! Host-side word grouping and id rectangles for the encoder and decoder stages.
//!
//! Both stages run one word per batch row, and a batch is a dense `[words, tmax]`
//! rectangle — so an ungrouped window pads every row out to its longest word (~4.5x
//! wasted rows on Rust source, in both FLOPs and VRAM). Grouping by byte length
//! removes that: the groups run strictly one after another, each is its own
//! rectangle, and their gradients accumulate, so cutting a window into groups changes
//! no arithmetic.
//!
//! Groups are **exact-length**, so a rectangle carries no padding at all and `tmax` is
//! the group's word length plus its closing `[W]` step. Measured over a 4096-word
//! window of Rust source: 16 groups, 0% padding, against 5 groups and ~20% padding
//! when lengths were bucketed by `next_power_of_two`.
//!
//! Everything here is index bookkeeping fed to the gather/scatter/CE kernels, plus
//! [`GroupIds`], which gets one group's lists onto the device in a single transfer.
//! Ids are built as `u32` (what the kernels take) so uploading them needs no narrowing
//! copy, and every buffer is allocated once at its window-worst-case size — a training
//! run rebuilds these per window forever, so nothing may allocate on the hot path.

use std::range::Range;

use cudarc::driver::{CudaEvent, CudaSlice, CudaView, PinnedHostSlice};

use super::Gpu;
use crate::config::{MAX_WORD_BYTES, WORDS_PER_SEQ};

/// Tallest rectangle a group can have: a word's bytes plus its closing `[W]` step.
const MAX_TMAX: usize = MAX_WORD_BYTES + 1;

/// Slots (rectangle cells) one window can need. Groups never pad, so a word takes
/// exactly `len + 1` slots.
const MAX_SLOTS: usize = WORDS_PER_SEQ * MAX_TMAX;

/// Elements one [`GroupIds`] upload can ever need, so its staging
/// buffers are allocated once at this size and never grow: the decoder's six lists for
/// one group, at the largest rectangle a window can produce. Four are per-slot
/// (`char_rows`/`char_ids` hold one char per slot bar each word's first), two are
/// per-word. An uncapped config puts the whole window in one group, so the bound is
/// the window's, not `max_rows`'.
pub const MAX_GROUP_IDS: usize = 4 * MAX_SLOTS + 2 * WORDS_PER_SEQ;

/// Rows the widest rectangle a group cap of `cap` allows (`0` = uncapped), so a
/// buffer sized to it fits every group of every window. A cap is a row budget that
/// converts to a word count, but a rectangle is never narrower than a single word,
/// so `tmax` is the floor and a cap below it does not shrink the rectangle further.
pub const fn max_group_rows(cap: usize) -> usize {
    if cap == 0 {
        MAX_SLOTS
    } else if cap < MAX_TMAX {
        MAX_TMAX
    } else {
        cap
    }
}

/// One rectangle: a run of same-length words, and where its cells live.
#[derive(Clone, Copy, Default)]
struct Group {
    /// First of this group's rows in [`Grouping::order`].
    first: u32,
    /// Words in the group — the rectangle's height in words, not in cells.
    words: u32,
    /// Rectangle width: the group's word length plus the `[W]` step.
    tmax: u32,
    /// First cell of this group's `[words, tmax]` rectangle in the flat arenas.
    slot: u32,
}

impl Group {
    fn rows(&self) -> usize {
        (self.words * self.tmax) as usize
    }

    fn word_span(&self) -> std::ops::Range<usize> {
        self.first as usize..(self.first + self.words) as usize
    }

    fn slot_span(&self) -> std::ops::Range<usize> {
        self.slot as usize..self.slot as usize + self.rows()
    }
}

/// A window's words ordered by length and cut into rectangles.
struct Grouping {
    /// Window word index per rectangle row, ascending by word length.
    order: Vec<u32>,
    groups: Vec<Group>,
    /// Cells across every group.
    slots: usize,
}

impl Grouping {
    fn new() -> Self {
        Self {
            order: vec![0; WORDS_PER_SEQ],
            // A cap of one row per group puts every word in its own rectangle.
            groups: Vec::with_capacity(WORDS_PER_SEQ),
            slots: 0,
        }
    }

    /// `max_rows` is the largest rectangle a group may reach before it is split into
    /// equal pieces, or `0` for no cap. See [`crate::config::GROUP_MAX_ROWS`].
    fn build(&mut self, lens: &[usize], max_rows: usize) {
        assert!(
            lens.len() <= WORDS_PER_SEQ,
            "window of {} words exceeds WORDS_PER_SEQ ({WORDS_PER_SEQ})",
            lens.len()
        );
        self.groups.clear();

        // Counting sort by exact length: count at `len + 1`, prefix-sum, then place.
        // Ascending length is also the group order, which fixes the order the
        // per-group losses are summed and the grads are scattered.
        // Words per length, prefix-summed into `order` offsets.
        let mut starts = [0u32; MAX_TMAX + 1];
        for &l in lens {
            starts[l + 1] += 1;
        }
        for l in 1..starts.len() {
            starts[l] += starts[l - 1];
        }
        let mut cursor = starts;
        for (w, &l) in lens.iter().enumerate() {
            self.order[cursor[l] as usize] = w as u32;
            cursor[l] += 1;
        }

        self.slots = 0;
        for len in 1..=MAX_WORD_BYTES {
            let (first, end) = (starts[len], starts[len + 1]);
            let words = end - first;
            if words == 0 {
                continue;
            }
            // `tmax` is fixed by the length, so a row cap converts to a word count.
            // Splitting into equal pieces rather than `per_piece` plus a remainder
            // keeps every rectangle wide — a 1-word tail launch is nearly pure
            // overhead and the pool sizes to the largest piece either way.
            let tmax = len as u32 + 1;
            let per_piece = match max_rows {
                0 => words,
                cap => (cap as u32 / tmax).max(1).min(words),
            };
            let piece = words.div_ceil(words.div_ceil(per_piece));

            let mut row = first;
            while row < end {
                let words = piece.min(end - row);
                self.groups.push(Group {
                    first: row,
                    words,
                    tmax,
                    slot: self.slots as u32,
                });
                self.slots += (words * tmax) as usize;
                row += words;
            }
        }
    }

    /// Window word index per row of group `g`.
    fn words(&self, g: usize) -> &[u32] {
        &self.order[self.groups[g].word_span()]
    }
}

/// A `[words, tmax]` encoder rectangle: the ids to embed and the rows to read `e_w`
/// from.
pub struct EncGroup<'a> {
    /// Window word index per row — where this group's `e_w` rows scatter back to.
    pub words: &'a [u32],
    /// Rectangle width, `word_len + 1`.
    pub tmax: usize,
    /// `[words, tmax]` token ids, padding slots at id 0.
    pub ids: &'a [u32],
}

/// Word `i`'s `[W]` step is rectangle row `(i + 1) * tmax - 1`: a group holds one word
/// length, so the readout row needs no table. See [`super::ops::pack_rows_u32`].

impl EncGroup<'_> {
    pub fn n_words(&self) -> usize {
        self.words.len()
    }

    pub fn rows(&self) -> usize {
        self.ids.len()
    }
}

/// Every encoder rectangle of one window.
///
/// Built once per window rather than once per direction: the encoder backward
/// re-forwards each group (activation checkpointing) and needs byte-identical ids.
pub struct EncoderGroups {
    grouping: Grouping,
    lens: Vec<usize>,
    ids: Vec<u32>,
}

impl EncoderGroups {
    pub fn new() -> Self {
        Self {
            grouping: Grouping::new(),
            lens: Vec::with_capacity(WORDS_PER_SEQ),
            ids: vec![0; MAX_SLOTS],
        }
    }

    /// Group `words[..dw]` by length and fill every group's id rectangle.
    pub fn build(
        &mut self,
        tokens: &[usize],
        words: &[Range<usize>],
        w_token: usize,
        max_rows: usize,
    ) {
        let dw = words.len() - 1;
        self.lens.clear();
        self.lens
            .extend((0..dw).map(|w| words[w].end - words[w].start));
        self.grouping.build(&self.lens, max_rows);

        // Padding slots stay at id 0 — they are masked out of the readout.
        self.ids[..self.grouping.slots].fill(0);
        for g in 0..self.len() {
            let grp = self.grouping.groups[g];
            let tmax = grp.tmax as usize;
            for (i, &w) in self.grouping.words(g).iter().enumerate() {
                let (start, len) = (words[w as usize].start, self.lens[w as usize]);
                let row = grp.slot as usize + i * tmax;
                for k in 0..len {
                    self.ids[row + k] = tokens[start + k] as u32;
                }
                self.ids[row + len] = w_token as u32;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.grouping.groups.len()
    }

    pub fn group(&self, g: usize) -> EncGroup<'_> {
        let grp = self.grouping.groups[g];
        EncGroup {
            words: self.grouping.words(g),
            tmax: grp.tmax as usize,
            ids: &self.ids[grp.slot_span()],
        }
    }

    /// Rows of the largest group — what the encoder's pool has to hold, since the
    /// groups run one after another and every buffer sizes to the widest one.
    pub fn rows_max(&self) -> usize {
        self.grouping
            .groups
            .iter()
            .map(Group::rows)
            .max()
            .unwrap_or(0)
    }
}

/// A `[words, tmax]` decoder rectangle and the six id lists addressing it.
///
/// A masked (prompt) word is still fed forward so its state and the tied-table char
/// grads are produced, but its slots get CE mask 0 — no loss, no logit gradient from
/// a prompt token.
pub struct DecGroup<'a> {
    /// Window word index per row.
    pub words: &'a [u32],
    pub tmax: usize,
    /// Word slot per row, for gathering the backbone context.
    pub word_ids: &'a [u32],
    /// Word `i`'s injected-context slot is rectangle row `i * tmax` — affine for the
    /// same reason as the encoder's readout.
    /// Rows fed by a previous char, and the char feeding each.
    pub char_rows: &'a [u32],
    pub char_ids: &'a [u32],
    /// CE target and loss mask per slot.
    pub targets: &'a [u32],
    pub mask: &'a [u32],
}

impl DecGroup<'_> {
    pub fn n_words(&self) -> usize {
        self.words.len()
    }

    pub fn rows(&self) -> usize {
        self.targets.len()
    }

    pub fn n_chars(&self) -> usize {
        self.char_rows.len()
    }
}

/// Every decoder rectangle of one window.
pub struct DecoderGroups {
    grouping: Grouping,
    lens: Vec<usize>,
    word_ids: Vec<u32>,
    char_rows: Vec<u32>,
    char_ids: Vec<u32>,
    /// Group `g`'s first char in `char_rows` / `char_ids`. A group holds one char per
    /// slot except each word's first, so under padding the count is not derivable
    /// from the rectangle alone.
    char_start: Vec<u32>,
    targets: Vec<u32>,
    mask: Vec<u32>,
    valid_rows: usize,
    scored_words: usize,
}

impl DecoderGroups {
    pub fn new() -> Self {
        Self {
            grouping: Grouping::new(),
            lens: Vec::with_capacity(WORDS_PER_SEQ),
            word_ids: vec![0; WORDS_PER_SEQ],
            char_rows: vec![0; MAX_SLOTS],
            char_ids: vec![0; MAX_SLOTS],
            char_start: Vec::with_capacity(WORDS_PER_SEQ + 1),
            targets: vec![0; MAX_SLOTS],
            mask: vec![0; MAX_SLOTS],
            valid_rows: 0,
            scored_words: 0,
        }
    }

    /// Group the decoded words (`words[w + 1]`, the word each backbone context
    /// predicts) by length and fill every group's rectangle.
    pub fn build(
        &mut self,
        tokens: &[usize],
        words: &[Range<usize>],
        w_token: usize,
        word_on: &impl Fn(usize) -> bool,
        max_rows: usize,
    ) {
        let dw = words.len() - 1;
        self.lens.clear();
        self.lens
            .extend((0..dw).map(|w| words[w + 1].end - words[w + 1].start));
        self.grouping.build(&self.lens, max_rows);

        // Every group scales by the WINDOW's valid-row count, so the summed loss and
        // grads match what one big rectangle would have produced. Under SFT masking
        // only the response words' rows count.
        self.valid_rows = (0..dw)
            .filter(|&w| word_on(w))
            .map(|w| self.lens[w] + 1)
            .sum();
        self.scored_words = (0..dw).filter(|&w| word_on(w)).count();

        // Padding slots get target 0 and mask 0 — fed forward, never scored.
        self.targets[..self.grouping.slots].fill(0);
        self.mask[..self.grouping.slots].fill(0);
        self.char_start.clear();
        self.char_start.push(0);
        let mut chars = 0;
        for g in 0..self.len() {
            let grp = self.grouping.groups[g];
            let tmax = grp.tmax as usize;
            for (i, &w) in self.grouping.words(g).iter().enumerate() {
                let w = w as usize;
                let (start, len) = (words[w + 1].start, self.lens[w]);
                let on = word_on(w) as u32;
                let row = grp.slot as usize + i * tmax;

                self.word_ids[grp.first as usize + i] = w as u32;
                for k in 1..=len {
                    self.char_rows[chars] = (i * tmax + k) as u32;
                    self.char_ids[chars] = tokens[start + k - 1] as u32;
                    chars += 1;
                }
                for k in 0..len {
                    self.targets[row + k] = tokens[start + k] as u32;
                    self.mask[row + k] = on;
                }
                self.targets[row + len] = w_token as u32;
                self.mask[row + len] = on;
            }
            self.char_start.push(chars as u32);
        }
    }

    pub fn len(&self) -> usize {
        self.grouping.groups.len()
    }

    pub fn group(&self, g: usize) -> DecGroup<'_> {
        let grp = self.grouping.groups[g];
        let chars = self.char_start[g] as usize..self.char_start[g + 1] as usize;
        DecGroup {
            words: self.grouping.words(g),
            tmax: grp.tmax as usize,
            word_ids: &self.word_ids[grp.word_span()],
            char_rows: &self.char_rows[chars.clone()],
            char_ids: &self.char_ids[chars],
            targets: &self.targets[grp.slot_span()],
            mask: &self.mask[grp.slot_span()],
        }
    }

    /// Rows of the largest group — what the decoder's pool has to hold.
    pub fn rows_max(&self) -> usize {
        self.grouping
            .groups
            .iter()
            .map(Group::rows)
            .max()
            .unwrap_or(0)
    }

    /// Scored slots and scored words of the window: the CE normalizer, and what the
    /// per-word loss metric re-averages over. Both drop masked (prompt) words.
    pub fn scored(&self) -> (usize, usize) {
        (self.valid_rows, self.scored_words)
    }
}

/// Staging slots the uploads rotate through. The per-slot event below is what makes
/// reuse safe; the slot count only decides how often the CPU reaches that wait.
/// Measured at 2, 4 and 64: no difference in step time, so two is enough.
const SLOTS: usize = 2;

/// One group's id lists, uploaded to the device in a single async transfer.
///
/// A group's bookkeeping — which row is a `[W]` step, which slot is a char, the CE
/// targets and mask — is a handful of small `u32` lists. Sending each with its own
/// [`super::ops::upload_ids_u32`] costs a device allocation and a *blocking* H2D
/// apiece (`memcpy_htod` from a pageable host slice is synchronous, so the compute
/// stream stalls on every one). The decoder issues six lists per group and the encoder
/// three, both again in backward: ~30 stalls per window for ~100 KB of data.
///
/// [`upload`](Self::upload) instead packs the lists back to back into one pinned host
/// buffer and issues one async copy, which overlaps the compute already queued.
/// [`list`](Self::list) hands back a device view per list, indexed by position in the
/// call. It borrows `&self`, so the borrow checker keeps a view from outliving the
/// next upload.
///
/// Every buffer is allocated once at [`MAX_GROUP_IDS`], the most any group of any
/// window can need, so no upload allocates or page-locks.
///
/// # Reuse hazard
///
/// The copy is async and reads the *pinned* buffer, while the refill at the top of
/// `upload` is a plain CPU write — so a single buffer is overwritten by the next group
/// while the previous group's copy, and the kernels reading its views, are still
/// outstanding. The ids silently become another group's: training diverged (loss
/// 2.6 -> 69) with no error anywhere.
///
/// Rotating slots alone does **not** fix that, which is the trap: it buys `SLOTS - 1`
/// iterations of slack, and the CPU runs arbitrarily far ahead of the device inside a
/// loop over groups, so it laps the rotation. What makes it safe is `copied` — the CPU
/// blocks on a slot's previous copy before refilling it.
pub struct GroupIds {
    host: [PinnedHostSlice<u32>; SLOTS],
    dev: [CudaSlice<u32>; SLOTS],
    /// Slot the last `upload` wrote, and the one `list` therefore reads.
    cur: usize,
    /// Completion of each slot's H2D copy — the CPU waits on it before refilling.
    copied: [Option<CudaEvent>; SLOTS],
    /// `(offset, len)` of each list of the current upload, in elements.
    spans: Vec<(usize, usize)>,
}

impl GroupIds {
    /// Device bytes held for the whole run, and the same again in pinned host memory:
    /// [`SLOTS`] slots of [`MAX_GROUP_IDS`] ids. 1.1 MiB per slot at the current config.
    pub const BYTES_PER_SIDE: usize = SLOTS * MAX_GROUP_IDS * size_of::<u32>();

    pub fn new(gpu: &Gpu) -> Self {
        Self {
            // SAFETY: uninitialised staging memory. An upload copies out only the
            // prefix it just wrote, and only the spans of that copy are ever read.
            host: std::array::from_fn(|_| {
                unsafe { gpu.context.alloc_pinned::<u32>(MAX_GROUP_IDS) }
                    .expect("GroupIds: pinned alloc")
            }),
            dev: std::array::from_fn(|_| {
                unsafe { gpu.stream.alloc::<u32>(MAX_GROUP_IDS) }.expect("GroupIds: alloc")
            }),
            copied: [const { None }; SLOTS],
            cur: 0,
            spans: Vec::with_capacity(8),
        }
    }

    /// Pack `lists` into one upload. Returns immediately; the copy is queued on
    /// `gpu.stream`, so the views are safe for any kernel launched after this.
    pub fn upload(&mut self, gpu: &Gpu, lists: &[&[u32]]) {
        self.spans.clear();
        let mut total = 0;
        for l in lists {
            self.spans.push((total, l.len()));
            total += l.len();
        }
        if total == 0 {
            return;
        }
        assert!(
            total <= MAX_GROUP_IDS,
            "GroupIds: {total} ids over the fixed capacity {MAX_GROUP_IDS}"
        );

        self.cur = (self.cur + 1) % SLOTS;
        let slot = self.cur;
        // A device-side `stream.wait` would not do: the racing write below is the
        // CPU's, and the CPU is what has to be held back.
        if let Some(ev) = &self.copied[slot] {
            ev.synchronize().expect("GroupIds: wait for prior H2D");
        }

        let host = self.host[slot].as_mut_slice().expect("pinned host slice");
        for (l, &(off, len)) in lists.iter().zip(&self.spans) {
            host[off..off + len].copy_from_slice(l);
        }
        // Only the prefix just written: the tail still holds an earlier group's ids.
        gpu.stream
            .memcpy_htod(&host[..total], &mut self.dev[slot].slice_mut(..total))
            .expect("GroupIds: H2D");
        self.copied[slot] = Some(
            gpu.stream
                .record_event(super::host_wait_event_flags())
                .expect("GroupIds: record H2D completion"),
        );
    }

    /// The `i`-th list of the last upload, as a device view.
    pub fn list(&self, i: usize) -> CudaView<'_, u32> {
        let (off, len) = self.spans[i];
        self.dev[self.cur].slice(off..off + len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::gpu::{GTensor, ops};
    use crate::tensor::Tensor;

    /// Consecutive uploads must not corrupt each other's ids.
    ///
    /// The uploads are async, so a group whose slot is refilled too early has its ids
    /// replaced by a later group's — no error, just wrong gathers.
    ///
    /// **This test does not reproduce that failure.** Single-slotting `GroupIds`
    /// diverges real training deterministically (char loss 2.6 -> 69 within ~10 steps)
    /// while this test still passes, at every group count and contention level tried;
    /// the reuse window evidently needs the real workload's queue depth. It guards the
    /// packing and offset arithmetic, which it does cover — the slot rotation is pinned
    /// only by that training run, so treat a green suite here as insufficient evidence.
    #[test]
    fn uploads_do_not_clobber_each_other() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        // A table whose row `i` is all `i`, so a gathered row names the id that
        // fetched it and a stale id is obvious.
        let rows = 64;
        let table = GTensor::from_host(
            &gpu,
            &Tensor::new(
                &[rows, 4],
                (0..rows).flat_map(|i| [i as f32; 4]).collect::<Vec<_>>(),
            ),
        );

        let mut ids = GroupIds::new(&gpu);
        let mut got = Vec::new();
        // Lists of differing lengths, so every upload writes a different prefix.
        let groups: Vec<Vec<u32>> = (0..8)
            .map(|g| {
                (0..(3 + g * 5))
                    .map(|i| ((g * 7 + i) % rows) as u32)
                    .collect()
            })
            .collect();

        // Something slow between the upload and the gather, so the copy for group
        // `g+1` is issued while group `g`'s gather is still queued — the interleaving
        // real training produces and a bare upload/gather pair does not.
        let busy = GTensor::zeros(&gpu, &[512, 512]);
        let mut sink = GTensor::uninit(&gpu, &[512, 512]);

        for list in &groups {
            ids.upload(&gpu, &[list]);
            ops::matmul_nn_into(&gpu, &busy, &busy, &mut sink, 0.0);
            // Gather, keeping the result on the device — no sync here, so an unsound
            // reuse has every chance to show.
            got.push(ops::embedding_gather_u32(
                &gpu,
                &table,
                &ids.list(0),
                list.len(),
                4,
            ));
        }

        for (list, out) in groups.iter().zip(&got) {
            let host = out.to_host(&gpu).data;
            for (row, &id) in list.iter().enumerate() {
                assert_eq!(
                    host[row * 4],
                    id as f32,
                    "row {row} gathered id {} instead of {id} — a later upload clobbered it",
                    host[row * 4]
                );
            }
        }
    }

    /// Several lists in one upload must come back at their own offsets.
    #[test]
    fn upload_packs_multiple_lists() {
        let Some(gpu) = crate::gpu::test_gpu() else {
            return;
        };
        let mut ids = GroupIds::new(&gpu);
        let a: Vec<u32> = vec![5, 1, 9];
        let b: Vec<u32> = vec![7, 7];
        let c: Vec<u32> = vec![0, 3, 2, 8];
        ids.upload(&gpu, &[&a, &b, &c]);

        for (i, want) in [&a, &b, &c].iter().enumerate() {
            let host = gpu.stream.clone_dtoh(&ids.list(i)).expect("dtoh");
            assert_eq!(&host, *want, "list {i} came back wrong");
        }
    }

    fn rng() -> impl FnMut(usize) -> usize {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        move |n| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % n as u64) as usize
        }
    }

    /// The invariants the encoder and decoder rely on: every word lands in exactly one
    /// group, a group holds one length only (so its rectangle has no padding), groups
    /// run shortest first — group order fixes the order the per-group losses are summed
    /// and the grads are scattered — and no rectangle exceeds the row cap.
    #[test]
    fn groups_partition_by_exact_length() {
        let mut rng = rng();
        let mut grouping = Grouping::new();
        for case in 0..500 {
            let lens: Vec<usize> = (0..rng(200) + 1).map(|_| rng(MAX_WORD_BYTES) + 1).collect();
            let max_rows = [0, 4, 17, 64, 2048][case % 5];
            grouping.build(&lens, max_rows);

            let mut seen = vec![false; lens.len()];
            let mut prev_tmax = 0;
            for g in 0..grouping.groups.len() {
                let grp = grouping.groups[g];
                assert!(grp.tmax >= prev_tmax, "case {case}: groups not ascending");
                prev_tmax = grp.tmax;
                assert!(
                    max_rows == 0 || grp.rows() <= max_rows.max(grp.tmax as usize),
                    "case {case}: group of {} rows over cap {max_rows}",
                    grp.rows()
                );
                for &w in grouping.words(g) {
                    assert!(!seen[w as usize], "case {case}: word {w} in two groups");
                    seen[w as usize] = true;
                    assert_eq!(
                        lens[w as usize] + 1,
                        grp.tmax as usize,
                        "case {case}: mixed lengths in one group"
                    );
                }
            }
            assert!(seen.iter().all(|&s| s), "case {case}: word left out");
        }
    }

    /// A capped run is cut into equal pieces, not into full pieces plus a remainder:
    /// a 1-word tail launch is nearly pure overhead.
    #[test]
    fn cap_splits_evenly() {
        let mut grouping = Grouping::new();
        // 9 words of length 3 (tmax 4), cap 16 rows -> 4 words per piece -> 3 pieces
        // of 3, not 4 + 4 + 1.
        grouping.build(&[3; 9], 16);
        let sizes: Vec<u32> = grouping.groups.iter().map(|g| g.words).collect();
        assert_eq!(sizes, vec![3, 3, 3]);
    }
}
