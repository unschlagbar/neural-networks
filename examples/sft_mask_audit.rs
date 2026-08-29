// What an SFT corpus's loss mask actually covers, over the whole file: how much
// of it is scored, and the two ways the mask can silently lose a target — a
// word 0 flagged for loss (nothing decodes it) and a scored `<SEP>` (the model
// would learn to emit the separator as response text).
//
//   cargo run --release --example sft_mask_audit -- data/mix/assistant_qa.jsonl
use neural_networks::sft;
use neural_networks::tokenizer_utf8::{SEP_TOKEN, Utf8Tokenizer};

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let tok = Utf8Tokenizer::new();
    let ex = sft::load_jsonl(&tok, &path, 28672).unwrap();
    let mut words = 0usize;
    let mut loss_words = 0usize;
    let mut rows = 0usize;
    let mut loss_rows = 0usize;
    let mut first_word_loss = 0usize;
    let mut sep_carries_loss = 0usize;
    let mut result_carries_loss = 0usize;
    let mut with_masked_turn = 0usize;
    let mut masked_carries_loss = 0usize;
    for e in &ex {
        words += e.words.len();
        rows += e.tokens.len() + e.words.len();
        // A tool result is prompt-side: the model reads it, never writes it.
        // Scanning the scored words for the wrapper catches a mis-built mask
        // that the per-example tests cannot see.
        let scored: String = e
            .words
            .iter()
            .zip(&e.loss)
            .filter(|&(_, &k)| k)
            .map(|(w, _)| tok.to_text_markup(&e.tokens[w.start..w.end]))
            .collect();
        if scored.contains(sft::RESULT_OPEN) {
            result_carries_loss += 1;
        }
        for (i, w) in e.words.iter().enumerate() {
            if e.loss[i] {
                loss_words += 1;
                loss_rows += w.end - w.start + 1;
                // A separator must never be a scored target.
                if w.end - w.start == 1 && e.tokens[w.start] == SEP_TOKEN {
                    sep_carries_loss += 1;
                }
            }
        }
        if e.loss[0] {
            first_word_loss += 1;
        }
    }
    println!("examples {}", ex.len());
    println!("words {words}, loss words {loss_words} ({:.1}%)", 100.0 * loss_words as f64 / words as f64);
    println!("rows  {rows}, loss rows  {loss_rows} ({:.1}%)", 100.0 * loss_rows as f64 / rows as f64);
    println!("examples whose word 0 carries loss (unscored, would be silently lost): {first_word_loss}");
    println!("<SEP> words carrying loss: {sep_carries_loss}");
    println!("examples whose scored words include a <result> wrapper: {result_carries_loss}");
    // The masked-turn check needs the roles, which an SftExample no longer
    // carries, so it re-reads the file: for every `assistant_context` turn,
    // its text must be absent from the scored words of the example built from
    // the same conversation.
    for line in std::io::BufRead::lines(std::io::BufReader::new(
        std::fs::File::open(&path).unwrap(),
    )) {
        let line = line.unwrap();
        let Some(turns) = sft::parse_messages(&line) else {
            continue;
        };
        let masked: Vec<&str> = turns
            .iter()
            .filter(|t| t.role == sft::Role::AssistantContext)
            .map(|t| t.content.as_str())
            .collect();
        if masked.is_empty() {
            continue;
        }
        with_masked_turn += 1;
        let Some(e) = sft::build_example_turns(&tok, &turns) else {
            continue;
        };
        let scored: String = e
            .words
            .iter()
            .zip(&e.loss)
            .filter(|&(_, &k)| k)
            .map(|(w, _)| tok.to_text_markup(&e.tokens[w.start..w.end]))
            .collect();
        let whole = tok.to_text(&e.tokens);
        for m in masked {
            assert!(whole.contains(m), "a masked turn vanished from the context");
            if scored.contains(m) {
                masked_carries_loss += 1;
            }
        }
    }
    println!("examples carrying a masked assistant turn: {with_masked_turn}");
    println!("  of those, masked text found in the scored words: {masked_carries_loss}");
}
