// The byte tokenizer and the word split — the two definitions everything that
// touches a corpus has to agree on.

pub mod segment;
pub mod tokenizer_utf8;

/// Cap on the bytes of a single word (see [`segment`]). Words longer than this
/// — a giant identifier, a base64 blob, a huge indent block — are chopped into
/// pieces, which bounds the decoder's per-word unroll.
pub const MAX_WORD_BYTES: usize = 16;
