use crate::{Token, TokenInfo, TokenOcurrences, TokenizationError};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Debug, Default)]
pub struct BPETokenizer {
    // A map of word occurrences, where the key is the word and the value is the number of occurrences.
    word_ocurrences: HashMap<String, u64>,
    // A map of tokens, where key is a tuple of the left and right tokens, and the value is a TokenOcurrences struct that contains information about the occurrences of the token.
    tokens: HashMap<TokenInfo, TokenOcurrences>,
    // A map of word tokens, where the key is the word and the value is a vector of tuples containing the left token and an optional right token.
    word_tokens: HashMap<String, Vec<TokenInfo>>,
    // A max-heap that stores tuples of (count, left_token, right_token) for efficient retrieval of the most frequent tokens.
    max_heap: BinaryHeap<(u64, Reverse<TokenInfo>)>,
    // A map of tokens to their corresponding IDs.
    tokens_id: HashMap<Vec<u8>, u32>,
    // A map of IDs to their corresponding tokens.
    id_tokens: HashMap<u32, Vec<u8>>,
    // A map of vocabulary, where the key is the token ID and the value is a tuple of (left token ID, optional right token ID).
    vocabulary: Vec<Token>,
    // was initialized
    initialized: bool,
}

mod candidate_updates;
mod merges;
mod setup;

impl BPETokenizer {
    pub fn new(word_ocurrences: HashMap<String, u64>) -> Self {
        Self {
            word_ocurrences,
            tokens: HashMap::new(),
            word_tokens: HashMap::new(),
            max_heap: BinaryHeap::new(),
            tokens_id: HashMap::new(),
            id_tokens: HashMap::new(),
            vocabulary: Vec::new(),
            initialized: false,
        }
    }

    pub fn initialize(&mut self) -> Result<(), TokenizationError> {
        if self.initialized {
            return Err(TokenizationError::UnexpectedError(
                "Already initialized".to_string(),
            ));
        }
        self.setup();
        match self.quickoff() {
            Err(e) => {
                return Err(e);
            }
            Ok(_) => {}
        }
        self.initialized = true;
        Ok(())
    }
}

#[cfg(test)]
include!("bpe_tokenizer/tests.rs");
