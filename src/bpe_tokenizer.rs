use crate::{TokenInfo, TokenOcurrences, TokenizationError};

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Debug, Default)]
pub struct BPETokenizer {
    // Word registry: word id -> the word's bytes.
    words: Vec<Box<str>>,
    // Word id -> number of occurrences of the word in the corpus.
    word_freqs: Vec<u64>,
    // Word id -> the current token id sequence that forms the word.
    word_tokens: Vec<Vec<u32>>,
    // A map of candidate token pairs, where the key is the (left token id, right token id)
    // pair, and the value is a TokenOcurrences struct with the pair's occurrences.
    tokens: HashMap<(u32, u32), TokenOcurrences>,
    // A max-heap of candidate pairs by occurrence count, keyed as (count, pair bytes).
    // The pair bytes (via TokenInfo) break ties lexicographically, matching the
    // documented tie-break rule; they are the only place token bytes are stored
    // per candidate outside the registry.
    max_heap: BinaryHeap<(u64, Reverse<TokenInfo>)>,
    // A map of token bytes to their corresponding IDs.
    tokens_id: HashMap<Vec<u8>, u32>,
    // A map of IDs to their corresponding token bytes.
    id_tokens: HashMap<u32, Vec<u8>>,
    // Token id -> byte length of the token. Dense: indexed directly by token id.
    token_sizes: Vec<u32>,
    // A map of learned merges, where the key is the (left ID, right ID) pair and the value is the merged token ID.
    // Lower merged IDs were learned earlier, so they have merge priority during encoding.
    pair_merges: HashMap<(u32, u32), u32>,
    // was initialized
    initialized: bool,
}

mod candidate_updates;
mod encoding;
mod merges;
mod serialization;
mod setup;

impl BPETokenizer {
    /**
     * Builds a tokenizer over a corpus of words and their occurrence counts.
     * The words are interned into a registry; from this point on, all internal
     * structures reference words and tokens by id.
     */
    pub fn new(word_ocurrences: HashMap<String, u64>) -> Self {
        // Sort by word bytes so the registry assignment is deterministic.
        let mut sorted: Vec<(String, u64)> = word_ocurrences.into_iter().collect();
        sorted.sort();
        let words: Vec<Box<str>> = sorted
            .iter()
            .map(|(word, _)| Box::from(word.as_str()))
            .collect();
        let word_freqs: Vec<u64> = sorted.iter().map(|(_, count)| *count).collect();
        Self {
            words,
            word_freqs,
            word_tokens: Vec::new(),
            tokens: HashMap::new(),
            max_heap: BinaryHeap::new(),
            tokens_id: HashMap::new(),
            id_tokens: HashMap::new(),
            token_sizes: Vec::new(),
            pair_merges: HashMap::new(),
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
        self.quickoff()?;
        self.initialized = true;
        Ok(())
    }

    /**
     * Number of tokens known to the tokenizer: the 256 base byte tokens
     * plus every learned merge. This is the size the embedding table of an
     * LLM consuming this tokenizer must have.
     */
    pub fn vocab_size(&self) -> usize {
        self.id_tokens.len()
    }

    /**
     * Trains until the vocabulary reaches @target_vocab_size tokens,
     * or until no more merges are available in the corpus.
     */
    pub fn train(&mut self, target_vocab_size: usize) -> Result<(), TokenizationError> {
        if !self.initialized {
            return Err(TokenizationError::UnexpectedError(
                "The tokenizer must be initialized before training.".to_string(),
            ));
        }
        while self.vocab_size() < target_vocab_size {
            match self.next_token()? {
                Some(_) => {}
                None => break,
            }
        }
        Ok(())
    }

    /** The byte length of a registered token. */
    pub(super) fn token_size(&self, token_id: u32) -> u32 {
        self.token_sizes[token_id as usize]
    }

    /** The TokenInfo (pair bytes) of a registered (left id, right id) pair. */
    pub(super) fn pair_token_info(&self, pair: (u32, u32)) -> TokenInfo {
        TokenInfo::new(
            self.id_tokens[&pair.0].clone(),
            Some(self.id_tokens[&pair.1].clone()),
        )
    }

    /**
     * Resolves a TokenInfo (pair bytes) into its (left id, right id) pair,
     * or None if the bytes are not registered tokens.
     */
    pub(super) fn pair_of(&self, token_info: &TokenInfo) -> Option<(u32, u32)> {
        let left = self.tokens_id.get(token_info.get_left())?;
        let right = match token_info.get_right() {
            Some(right_bytes) => self.tokens_id.get(right_bytes)?,
            None => return None,
        };
        Some((*left, *right))
    }
}

#[cfg(test)]
include!("bpe_tokenizer/tests.rs");
