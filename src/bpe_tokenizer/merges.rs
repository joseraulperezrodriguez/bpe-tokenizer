use std::cmp::Reverse;
use std::collections::HashSet;

use crate::{TokenInfo, TokenizationError};

use super::BPETokenizer;

impl BPETokenizer {
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
                    // The heap can contain stale entries after a pair was removed,
                    // and unresolvable entries are skipped the same way.
                    let Some(pair) = self.pair_of(&token_info) else {
                        continue;
                    };
                    match self.tokens.get(&pair) {
                        Some(ocurrences) => {
                            if ocurrences.get_ocurrences_count() == count {
                                return Ok(Some(token_info));
                            }
                            if ocurrences.get_ocurrences_count() > 0 {
                                self.max_heap
                                    .push((ocurrences.get_ocurrences_count(), Reverse(token_info)));
                            } else {
                                self.tokens.remove(&pair);
                            }
                        }
                        None => {
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

    /**
     * It merges every occurrence of the winning pair in every word where it
     * appears, and generates new candidates from the tokens that became
     * adjacent after the merge.
     */
    pub(super) fn merge_winning_token_and_generate_new_ones(
        &mut self,
        pair: (u32, u32),
    ) -> Result<(), TokenizationError> {
        // The merged token id was registered in the vocabulary before this call.
        let winner_id = self
            .pair_merges
            .get(&pair)
            .copied()
            .ok_or_else(|| TokenizationError::MissingToken(String::from_utf8_lossy(
                &self.pair_token_info(pair).to_bytes(),
            ).into_owned()))?;

        // New candidate pairs collected while merging, pushed to the heap at the end.
        let mut new_pairs: HashSet<(u32, u32)> = HashSet::new();
        // Occurrences are collected first to avoid holding borrows while mutating.
        let mut word_occurrences: Vec<(u32, Vec<u32>)> = Vec::new();
        match self.tokens.get(&pair) {
            Some(pair_ocurrences) => {
                for (word_id, offsets) in pair_ocurrences.get_ocurrences() {
                    word_occurrences.push((*word_id, offsets.clone()));
                }
            }
            None => {
                return Err(TokenizationError::UnexpectedError(
                    "Unexpected non existing token.".to_string(),
                ));
            }
        }

        for (word_id, offsets) in word_occurrences {
            let word_freq = self.word_freqs.get(word_id as usize).copied().ok_or_else(|| {
                TokenizationError::UnexpectedError(format!(
                    "Unexpected missing word id {}.",
                    word_id
                ))
            })?;
            let tokens = self.word_tokens.get(word_id as usize).cloned().ok_or_else(|| {
                TokenizationError::UnexpectedError(format!(
                    "Unexpected missing word id {}.",
                    word_id
                ))
            })?;

            let mut index = 0;
            let mut byte_count: u32 = 0;
            let mut ocurrences_id = 0;
            let mut prev_new_token = false;

            let mut final_tokens: Vec<u32> = Vec::new();
            // (pair, word_id, word_freq, byte_offset) of candidates formed around the merges.
            let mut pending: Vec<((u32, u32), u32, u64, u32)> = Vec::new();

            while index < tokens.len() {
                // position the ocurrence_id at the first occurrence at or after the current token.
                while ocurrences_id < offsets.len() && byte_count > offsets[ocurrences_id] {
                    ocurrences_id += 1;
                }

                // at this point we know that we can merge tokens at index and index+1
                if ocurrences_id < offsets.len() && byte_count == offsets[ocurrences_id] {
                    // merging the winning pair parts into the winning token
                    if !final_tokens.is_empty() {
                        // if there is a previous token, we merge it with the winning token, resulting in a brand new candidate.
                        let left = final_tokens[final_tokens.len() - 1];
                        let left_size = self.token_size(left);
                        pending.push((
                            (left, winner_id),
                            word_id,
                            word_freq,
                            byte_count - left_size,
                        ));
                    }
                    final_tokens.push(winner_id);
                    prev_new_token = true;

                    // we advance 1 extra position here so next iteration falls at a new ocurrence, not at the second part of this ocurrence.
                    byte_count += self.token_size(tokens[index]);
                    index += 1;
                } else {
                    // This means the ocurrences of the tokens to merge is to the right of the current token index
                    if !final_tokens.is_empty() && prev_new_token {
                        let left = final_tokens[final_tokens.len() - 1];
                        let left_size = self.token_size(left);
                        pending.push((
                            (left, tokens[index]),
                            word_id,
                            word_freq,
                            byte_count - left_size,
                        ));
                    }
                    final_tokens.push(tokens[index]);
                    prev_new_token = false;
                }
                byte_count += self.token_size(tokens[index]);
                index += 1;
            }
            self.word_tokens[word_id as usize] = final_tokens;

            for (new_pair, word, freq, byte_offset) in pending {
                Self::insert_new_pair(
                    new_pair,
                    &mut new_pairs,
                    &mut self.tokens,
                    word,
                    freq,
                    byte_offset,
                );
            }
        }
        // Adding pairs that were created as part of the winning token merge to the max heap.
        for new_pair in new_pairs {
            self.push_pair_to_heap(new_pair)?;
        }
        Ok(())
    }

    /**
     * It gets the token pair with the most ocurrences, merges it, and returns the new token.
     * It also update the structures as follows:
     *  - Update all tokens that overlap with the winning pair.
     *  - Create new candidates from the ones that are adyancent to the winning token after merge it.
     *  - Register the winning token in the id maps and the merge table.
     */
    pub fn next_token(&mut self) -> Result<Option<(u32, TokenInfo)>, TokenizationError> {
        let Some(winning_info) = self.get_winning_token()? else {
            return Ok(None);
        };

        let Some(pair) = self.pair_of(&winning_info) else {
            return Err(TokenizationError::MissingToken(winning_info.to_string()));
        };

        // new token id and bytes for the winning pair.
        let token_id = self.id_tokens.len() as u32;
        let mut token_bytes = self.id_tokens[&pair.0].clone();
        token_bytes.extend_from_slice(&self.id_tokens[&pair.1]);

        // Adding the new token to the token id maps and the learned merge table.
        self.tokens_id.insert(token_bytes.clone(), token_id);
        self.id_tokens.insert(token_id, token_bytes.clone());
        self.token_sizes.push(token_bytes.len() as u32);
        self.pair_merges.insert(pair, token_id);

        // Update the overlapping candidates, removing the ocurrences that overlaps with the winning pair.
        let to_update = self.get_tokens_to_update(pair)?;
        self.update_tokens(&to_update)?;

        self.merge_winning_token_and_generate_new_ones(pair)?;

        Ok(Some((token_id, winning_info)))
    }
}
