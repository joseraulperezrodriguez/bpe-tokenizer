use std::collections::HashMap;

use bpe_tokenizer::{BPETokenizer, PreTokenization};

fn main() {
    // 1. Count pre-token words over a small corpus.
    let corpus = "the quick brown fox jumps over the lazy dog. the dog barks!";
    let mut word_ocurrences = HashMap::new();
    PreTokenization::count_tokens(corpus, &mut word_ocurrences);

    // 2. Train a small vocabulary.
    let mut tokenizer = BPETokenizer::new(word_ocurrences);
    tokenizer.initialize().unwrap();
    tokenizer.train(280).unwrap();
    println!("vocab size: {}", tokenizer.vocab_size());

    // 3. Encode and decode some text (including unseen words).
    for text in ["the quick brown fox", "the cat quacked"] {
        let ids = tokenizer.encode(text);
        let decoded = tokenizer.decode_to_string(&ids).unwrap();
        println!(
            "{:?} -> {} tokens {:?} -> {:?}",
            text,
            ids.len(),
            ids,
            decoded
        );
    }
}
