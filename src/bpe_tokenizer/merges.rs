use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use crate::{OcurrencePosition, Token, TokenInfo, TokenizationError};

use super::BPETokenizer;

impl BPETokenizer {
    /**
     * Given a token candidate, find which token ids it was formed from.
     * The left and right token must exists in the id registry.
     */
    pub(super) fn token_parts(
        &mut self,
        winning_token: &TokenInfo,
    ) -> Result<(u32, u32), TokenizationError> {
        let left_id = match self.tokens_id.get(winning_token.get_left()) {
            Some(left_id) => left_id.clone(),
            None => {
                return Err(TokenizationError::MissingToken(
                    String::from_utf8_lossy(winning_token.get_left()).into_owned(),
                ));
            }
        };
        let right_id = match winning_token.get_right() {
            Some(right_bytes) => {
                match self.tokens_id.get(right_bytes) {
                    Some(right_id) => right_id.clone(),
                    None => {
                        // At this point all tokens are composed by left and right.
                        return Err(TokenizationError::MissingToken(
                            String::from_utf8_lossy(right_bytes).into_owned(),
                        ));
                    }
                }
            }
            None => {
                return Err(TokenizationError::MissingToken(
                    winning_token.to_string().clone(),
                ));
            }
        };
        return Ok((left_id, right_id));
    }

    /**
     * Keep poping from the max heap until it finds an element count that matches what is in the
     * actual count.
     * A key assumption here is that the discrepancy between the count that we get from the queue
     * and the actual count, is because the actual count is lower (because we removed some ocurrences)
     * than the count we get from the queue.
     */
    pub fn get_winning_token(&mut self) -> Result<Option<TokenInfo>, TokenizationError> {
        while !self.max_heap.is_empty() {
            match self.max_heap.pop() {
                Some((count, Reverse(token_info))) => {
                    match self.tokens.get(&token_info) {
                        Some(tm) => {
                            if tm.get_ocurrences_count() == count {
                                return Ok(Some(token_info));
                            }
                            if tm.get_ocurrences_count() > 0 {
                                self.max_heap
                                    .push((tm.get_ocurrences_count(), Reverse(token_info)));
                            } else {
                                self.tokens.remove(&token_info);
                            }
                        }
                        None => {
                            // The heap can contain stale entries after a token was removed.
                            continue;
                        }
                    }
                }
                None => {
                    return Err(TokenizationError::UnexpectedError(
                        "Unexpected empty heap.".to_string(),
                    ))
                }
            }
        }
        Ok(None)
    }

    pub(super) fn merge_winning_token_and_generate_new_ones(
        &mut self,
        winning_token: &TokenInfo,
    ) -> Result<(), TokenizationError> {
        // Update the word tokens and generate new token from winning token adyacent ones.
        let mut new_tokens: HashSet<TokenInfo> = HashSet::new();
        let mut pending_tokens = Vec::new();
        match self.tokens.get(&winning_token) {
            Some(winning_token_ocurrences) => {
                // iterating over all words in which the winning token is present.
                for (word, wining_token_ocurrences_positions) in
                    winning_token_ocurrences.get_ocurrences()
                {
                    let word_ocurrences = match self.word_ocurrences.get(word) {
                        Some(wo) => wo.clone(),
                        None => {
                            return Err(TokenizationError::WordNotFound(word.to_string()));
                        }
                    };
                    match self.word_tokens.get(word) {
                        Some(tokens) => {
                            let mut index = 0;
                            let mut byte_count = 0;
                            let mut ocurrences_id = 0;
                            let mut prev_new_token = false;

                            let mut final_tokens: Vec<TokenInfo> = Vec::new();

                            while index < tokens.len() {
                                // position the ocurrence_id at the same index of the token id.
                                while ocurrences_id < wining_token_ocurrences_positions.len()
                                    && &byte_count
                                        > wining_token_ocurrences_positions[ocurrences_id]
                                            .0
                                            .get_byte_offset()
                                {
                                    ocurrences_id += 1;
                                }

                                // at this point we know that we can merge tokens at index and index+1
                                if ocurrences_id < wining_token_ocurrences_positions.len()
                                    && &byte_count
                                        == wining_token_ocurrences_positions[ocurrences_id]
                                            .0
                                            .get_byte_offset()
                                {
                                    // merging winning tokens part into one token
                                    let new_token = TokenInfo::new(
                                        tokens[index].to_bytes(),
                                        Some(tokens[index + 1].to_bytes()),
                                    );
                                    if final_tokens.len() > 0 {
                                        // if there is a previous token, we merge it with the winning token, resulting in a branch new token.
                                        let left = final_tokens[final_tokens.len() - 1].clone();
                                        // position where this new token appears.
                                        let position = (
                                            OcurrencePosition::new(
                                                byte_count - left.size(),
                                                left.size(),
                                            ),
                                            Some(OcurrencePosition::new(
                                                byte_count,
                                                new_token.size(),
                                            )),
                                        );
                                        pending_tokens.push((
                                            left,
                                            new_token.clone(),
                                            word.clone(),
                                            word_ocurrences,
                                            position,
                                        ));
                                    }
                                    final_tokens.push(new_token);
                                    prev_new_token = true;

                                    // we advance 1 extra position here so next iteration falls at a new ocurrence, not at the second part of this ocurrence.
                                    byte_count += tokens[index].size();
                                    index += 1;
                                } else {
                                    // This means the ocurrences of the tokens to merge is to the right of the current token index
                                    if final_tokens.len() > 0 && prev_new_token {
                                        let left = final_tokens[final_tokens.len() - 1].clone();
                                        let position = (
                                            OcurrencePosition::new(
                                                byte_count - left.size(),
                                                left.size(),
                                            ),
                                            Some(OcurrencePosition::new(
                                                byte_count,
                                                tokens[index].size(),
                                            )),
                                        );
                                        pending_tokens.push((
                                            left,
                                            tokens[index].clone(),
                                            word.clone(),
                                            word_ocurrences,
                                            position,
                                        ));
                                    }
                                    final_tokens.push(tokens[index].clone());
                                    prev_new_token = false;
                                }
                                byte_count += tokens[index].size();
                                index += 1;
                            }
                            self.word_tokens.insert(word.clone(), final_tokens);
                        }
                        None => {
                            return Err(TokenizationError::UnexpectedError(
                                "Unexpected missing word".to_string(),
                            ));
                        }
                    }
                }
            }
            None => {
                return Err(TokenizationError::UnexpectedError(
                    "Unexpected non existing token.".to_string(),
                ));
            }
        }
        // Insert all pending tokens.
        for (left, right, word, word_ocurrences, position) in pending_tokens {
            Self::insert_new_token(
                &left,
                &right,
                &mut new_tokens,
                &mut self.tokens,
                &word,
                word_ocurrences,
                position,
            );
        }
        // Adding tokens that were created as part of the winning token merge to the max heap.
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

    /**
     * It gets the token with the most ocurrences and return it.
     * It also update the structures as follows:
     *  - Update all tokens that overlap with the winning token.
     *  - Create new tokens from the ones that are adyancent to the winning token after merge it.
     *  - Register the winning token in the vocabulary.
     */
    pub fn next_token(&mut self) -> Result<Option<(u32, TokenInfo)>, TokenizationError> {
        let winning_token = match self.get_winning_token() {
            Ok(wt) => wt,
            Err(e) => {
                return Err(e);
            }
        };

        let Some(winning_token) = winning_token else {
            return Ok(None);
        };

        // new token id for the winning token.
        let token_id = self.vocabulary.len() as u32;
        // getting the left and right ids that compounds the new token.
        let (left_id, right_id) = match self.token_parts(&winning_token) {
            Ok((left_id, right_id)) => (left_id, right_id),
            Err(e) => {
                return Err(e);
            }
        };

        // Adding the new token to the vocabulary and the token id maps.
        self.vocabulary
            .push(Token::new(token_id, left_id, Some(right_id)));
        self.tokens_id.insert(winning_token.to_bytes(), token_id);
        self.id_tokens.insert(token_id, winning_token.to_bytes());

        // It holds the positions that we need to remove from the ocurrences list in a given token for a a given word.
        // Key: The token candidate that we need to update
        // Value: a word -> token list indexes map, representing which ocurrences in the token list for the word must be removed.
        let to_update: HashMap<
            TokenInfo,
            HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>,
        > = match self.get_tokens_to_update(&winning_token) {
            Ok(tu) => tu,
            Err(e) => {
                return Err(e);
            }
        };
        // Update the overlapping tokens, removing the ocurrences that overlaps with the winning token.
        match self.update_tokens(&to_update) {
            Ok(_) => (),
            Err(e) => {
                return Err(e);
            }
        }

        match self.merge_winning_token_and_generate_new_ones(&winning_token) {
            Ok(_) => {}
            Err(e) => {
                return Err(e);
            }
        }

        Ok(Some((token_id, winning_token)))
    }
}
