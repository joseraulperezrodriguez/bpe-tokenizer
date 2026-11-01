use std::collections::HashMap;

use bpe_tokenizer::{BPETokenizer, PreTokenization, TokenizationError};

// Trains a tokenizer on the given corpus for @merge_count merges.
fn trained_tokenizer(corpus: &str, merge_count: usize) -> BPETokenizer {
    let mut word_ocurrences = HashMap::new();
    PreTokenization::count_tokens(corpus, &mut word_ocurrences);
    let mut tokenizer = BPETokenizer::new(word_ocurrences);
    tokenizer.initialize().unwrap();
    for _ in 0..merge_count {
        tokenizer.next_token().unwrap();
    }
    tokenizer
}

#[test]
fn test_encode_basic() {
    let tokenizer = trained_tokenizer("the quick brown fox jumps over the lazy dog", 20);

    let text = "the quick brown fox";
    let ids = tokenizer.encode(text);

    // Encoding must be lossless and should compress below the raw byte count.
    let decoded = tokenizer.decode_to_string(&ids).unwrap();
    assert_eq!(decoded, text);
    assert!(ids.len() < text.len());

    // Learned tokens should appear in the encoding.
    let max_byte_id = 255u32;
    assert!(ids.iter().any(|id| *id > max_byte_id));
}

#[test]
fn test_decode_basic() {
    let tokenizer = trained_tokenizer("hello world hello", 10);

    // Round trip through bytes and string stays exact.
    let ids = tokenizer.encode("hello world");
    assert_eq!(tokenizer.decode(&ids).unwrap(), b"hello world".to_vec());
    assert_eq!(tokenizer.decode_to_string(&ids).unwrap(), "hello world");

    // Unknown ids are rejected with a dedicated error.
    let err = tokenizer.decode(&[1, 500_000]).unwrap_err();
    assert!(matches!(err, TokenizationError::UnknownTokenId(500_000)));
}
