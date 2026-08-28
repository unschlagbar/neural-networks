// The byte tokenizer and the word split — the two definitions everything that
// touches a corpus has to agree on.
//
// They live in their own crate because both sides of the pipeline need them:
// the model (`neural-networks`) trains on words, and the corpus builder
// (`datamix`) has to count and cap the same words the trainer will see. Sharing
// the crate rather than the convention is what makes "4096 words" mean one
// thing. `neural-networks` re-exports both modules, so `crate::segment::…` and
// `crate::tokenizer_utf8::…` still resolve inside it.

pub mod segment;
pub mod tokenizer_utf8;

/// Cap on the bytes of a single word (see [`segment`]). Words longer than this
/// — a giant identifier, a base64 blob, a huge indent block — are chopped into
/// pieces, which bounds the decoder's per-word unroll.
///
/// It lives here rather than in the model's config because it is part of the
/// split itself: change it and the same text becomes different words.
pub const MAX_WORD_BYTES: usize = 16;
