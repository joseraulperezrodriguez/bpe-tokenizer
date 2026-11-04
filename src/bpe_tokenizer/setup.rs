use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use crate::{TokenInfo, TokenOcurrences, TokenizationError};

use super::BPETokenizer;

impl BPETokenizer {
    /**
     * It initializes the structures of the tokenizer.
     * Registers the base 256 byte tokens and builds, for every word in the
     * corpus, its initial token sequence: one token id per UTF-8 byte.
     */
    pub(super) fn setup(&mut self) {
        for byte in u8::MIN..=u8::MAX {
            let token = vec![byte];
            let token_id = u32::from(byte);
            self.tokens_id.insert(token.clone(), token_id);
            self.id_tokens.insert(token_id, token);
            self.token_sizes.push(1);
        }

        self.word_tokens.clear();
        for word_id in 0..self.words.len() {
            let mut tokens: Vec<u32> = Vec::new();
            if self.word_freqs[word_id] > 0 {
                for byte in self.words[word_id].bytes() {
                    tokens.push(u32::from(byte));
                }
            }
            self.word_tokens.push(tokens);
        }
    }

    /**
     * Utility method to register an occurrence of a new candidate pair.
     * A pair that is not in the tokens map yet is marked in @new_pairs so
     * the caller can push it to the heap.
     */
    pub(super) fn insert_new_pair(
        pair: (u32, u32),
        new_pairs: &mut HashSet<(u32, u32)>,
        tokens: &mut HashMap<(u32, u32), TokenOcurrences>,
        word_id: u32,
        count: u64,
        byte_offset: u32,
    ) {
        if !tokens.contains_key(&pair) {
            new_pairs.insert(pair);
        }
        tokens
            .entry(pair)
            .or_insert_with(TokenOcurrences::new)
            .add_ocurrence(word_id, count, byte_offset);
    }

    /**
     * It merges the initial tokens and prepare for full token iterations:
     * for every word, every adjacent token pair becomes a candidate.
     */
    pub(super) fn quickoff(&mut self) -> Result<(), TokenizationError> {
        let mut new_pairs: HashSet<(u32, u32)> = HashSet::new();
        for word_id in 0..self.word_tokens.len() {
            if self.word_freqs[word_id] == 0 {
                continue;
            }
            let tokens = &self.word_tokens[word_id];
            if tokens.is_empty() {
                return Err(TokenizationError::UnexpectedError(format!(
                    "Empty tokens list in the word {:?}",
                    self.words[word_id]
                )));
            }
            let count = self.word_freqs[word_id];
            let mut byte_index: u32 = 0;
            // iterates over all tokens that forms the current word and creates new token candidates by merging adyacent tokens.
            for index in 1..tokens.len() {
                let pair = (tokens[index - 1], tokens[index]);
                Self::insert_new_pair(
                    pair,
                    &mut new_pairs,
                    &mut self.tokens,
                    word_id as u32,
                    count,
                    byte_index,
                );
                byte_index += self.token_size(tokens[index - 1]);
            }
        }
        // Insert the initial set of candidates into the tokens heap.
        for pair in new_pairs {
            self.push_pair_to_heap(pair)?;
        }
        Ok(())
    }

    /**
     * Pushes a candidate pair to the heap with its current occurrence count,
     * keyed by the pair's bytes so ties are broken lexicographically.
     */
    pub(super) fn push_pair_to_heap(&mut self, pair: (u32, u32)) -> Result<(), TokenizationError> {
        let token_info: TokenInfo = self.pair_token_info(pair);
        let token_ocurrences = self
            .tokens
            .get(&pair)
            .ok_or_else(|| TokenizationError::MissingToken(token_info.to_string()))?;
        self.max_heap
            .push((token_ocurrences.get_ocurrences_count(), Reverse(token_info)));
        Ok(())
    }
}
