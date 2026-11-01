use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use crate::{OcurrencePosition, TokenInfo, TokenOcurrences, TokenizationError};

use super::BPETokenizer;

impl BPETokenizer {
    /**
     * It initializes the structures of the tokenizer.
     * Iterates over all words in the corpus, and build the initial tokens list.
     */
    pub(super) fn setup(&mut self) {
        for byte in u8::MIN..=u8::MAX {
            let token = vec![byte];
            let token_id = u32::from(byte);
            self.tokens_id.insert(token.clone(), token_id);
            self.id_tokens.insert(token_id, token.clone());
        }

        // Iterating over all words
        for (word, count) in &self.word_ocurrences {
            if *count == 0 {
                continue;
            }
            // List of tokens that initially form this word, one byte each.
            let mut word_tokens: Vec<TokenInfo> = Vec::new();
            for (index, byte) in word.bytes().enumerate() {
                let token = TokenInfo::new(vec![byte], None);
                word_tokens.push(token.clone());

                let position = (OcurrencePosition::new(index, 1), None);
                self.tokens
                    .entry(token)
                    .or_insert_with(TokenOcurrences::new)
                    .add_ocurrence(word.clone(), *count, position);
            }
            self.word_tokens.insert(word.clone(), word_tokens);
        }
    }

    /**
     * Utility method to add a new token to the tokens list.
     */
    pub(super) fn insert_new_token(
        left: &TokenInfo,
        right: &TokenInfo,
        new_tokens: &mut HashSet<TokenInfo>,
        tokens: &mut HashMap<TokenInfo, TokenOcurrences>,
        word: &String,
        count: u64,
        position: (OcurrencePosition, Option<OcurrencePosition>),
    ) {
        let candidate_token: TokenInfo = TokenInfo::merge(left, right);
        // inserting the ocurrence in the new tokens map if it is not an existing token.
        if !tokens.contains_key(&candidate_token) {
            new_tokens.insert(candidate_token.clone());
        }

        tokens
            .entry(candidate_token)
            .or_insert_with(TokenOcurrences::new)
            .add_ocurrence(word.clone(), count, position);
    }

    /**
     * It merges the initial tokens and prepare for full token iterations.
     */
    pub(super) fn quickoff(&mut self) -> Result<(), TokenizationError> {
        let mut new_tokens: HashSet<TokenInfo> = HashSet::new();
        // Iterate over all word tokens.
        for (word, word_tokens) in self.word_tokens.iter_mut() {
            // Get the number of ocurrences.
            let count = match self.word_ocurrences.get(word) {
                Some(&ocurrences) => ocurrences,
                None => {
                    return Err(TokenizationError::WordNotFound(word.clone()));
                }
            };

            if word_tokens.is_empty() {
                return Err(TokenizationError::UnexpectedError(
                    "Empty tokens list in the word {word}".to_string(),
                ));
            }
            let mut index: usize = 1;
            let mut byte_index: usize = 0;
            // iterates over all tokens that forms the current word and creates new token candidates by merging adyacent tokens.
            while index < word_tokens.len() {
                let prev: &TokenInfo = &word_tokens[index as usize - 1];
                let current: &TokenInfo = &word_tokens[index as usize];
                let position = (
                    OcurrencePosition::new(byte_index, prev.size()),
                    Some(OcurrencePosition::new(
                        byte_index + prev.size(),
                        current.size(),
                    )),
                );
                Self::insert_new_token(
                    prev,
                    current,
                    &mut new_tokens,
                    &mut self.tokens,
                    word,
                    count,
                    position,
                );

                index += 1;
                byte_index += prev.size();
            }
        }
        // Insert the initial set of tokens into the tokens heap.
        for token_info in new_tokens {
            match self.tokens.get(&token_info) {
                Some(token_ocurrences) => {
                    self.max_heap
                        .push((token_ocurrences.get_ocurrences_count(), Reverse(token_info)));
                }
                None => {
                    return Err(TokenizationError::MissingToken(token_info.to_string()));
                }
            }
        }
        Ok(())
    }
}
