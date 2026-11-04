use std::collections::{HashMap, HashSet};

use crate::TokenizationError;

use super::BPETokenizer;

impl BPETokenizer {
    /**
     * It find and constructs all candidate pairs that overlap on the left or
     * right with the pair's occurrences at @offsets.
     *
     * @map, where we store the candidate occurrences to remove.
     * @winning_pair, the pair that is about to be merged.
     * @word_id, the id of the word where we are reconstructing candidates.
     * @word_tokens, the current token sequence of the word.
     * @offsets, the ascending byte offsets where the winning pair occurs.
     */
    pub(super) fn find_and_insert_adyacent_tokens(
        &self,
        map: &mut HashMap<(u32, u32), HashMap<u32, Vec<u32>>>,
        winning_pair: (u32, u32),
        word_id: u32,
        word_tokens: &[u32],
        offsets: &[u32],
    ) -> Result<(), TokenizationError> {
        // Byte length of the winning pair's left part; occurrence offsets are
        // the offset of the left part.
        let left_size = self.token_size(winning_pair.0);

        let mut token_index_left = 0;
        let mut token_index_cumul_left: u32 = 0;

        let mut token_index_right = 0;
        let mut token_index_cumul_right: u32 = 0;

        let mut offsets_index = 0;
        let mut prev_skipped = false;
        // The pair between two adjacent occurrences of the winner is
        // reconstructed once per side; the duplicate is skipped.
        let mut last_inserted: Option<((u32, u32), u32)> = None;

        // We iterate over all offsets and form the left and right adyacent candidates.
        while offsets_index < offsets.len() {
            let offset = offsets[offsets_index];

            // If the previous occurrence overlaps with this one, we move one position forward,
            // because the current one won't be merged as a new token, so we don't need to find
            // its adyacent nodes to update.
            if offsets_index > 0 {
                let prev_offset = offsets[offsets_index - 1];
                if offset == prev_offset + left_size && !prev_skipped {
                    offsets_index += 1;
                    prev_skipped = true;
                    continue;
                }
            }
            prev_skipped = false;

            // Move @token_index_left pointer, until it points to the token that starts at @offset.
            while offset > token_index_cumul_left {
                token_index_cumul_left += self.token_size(word_tokens[token_index_left]);
                token_index_left += 1;
            }
            // at this point it is guaranteed that @token_index_left < word_tokens.len(),
            // because if @token_index_left points to the left part, then there needs to be another index that points at the right part
            // because @offsets is built from the token list in the end.

            // if there is a token before the left part, the pair it forms with the
            // left part is destroyed by the merge.
            if token_index_left > 0 {
                let candidate = (word_tokens[token_index_left - 1], word_tokens[token_index_left]);
                if candidate != winning_pair {
                    let candidate_offset =
                        offset - self.token_size(word_tokens[token_index_left - 1]);
                    if last_inserted != Some((candidate, candidate_offset)) {
                        map.entry(candidate)
                            .or_default()
                            .entry(word_id)
                            .or_default()
                            .push(candidate_offset);
                        last_inserted = Some((candidate, candidate_offset));
                    }
                }
            }

            // The right part of the pair starts at @offset + left_size.
            let right_start = offset + left_size;
            // Move @token_index_right pointer, until it points to the token that starts at @right_start.
            while right_start > token_index_cumul_right {
                token_index_cumul_right += self.token_size(word_tokens[token_index_right]);
                token_index_right += 1;
            }

            // only in this case, we confirm there is a token after the right part
            if token_index_right + 1 < word_tokens.len() {
                let candidate =
                    (word_tokens[token_index_right], word_tokens[token_index_right + 1]);
                if candidate != winning_pair {
                    if last_inserted != Some((candidate, right_start)) {
                        map.entry(candidate)
                            .or_default()
                            .entry(word_id)
                            .or_default()
                            .push(right_start);
                        last_inserted = Some((candidate, right_start));
                    }
                }
            }

            offsets_index += 1;
        }

        Ok(())
    }

    /**
     * Given a @winning_pair, it searches over all words and token ocurrences of it and find the candidates that needs
     * to be removed because overlaps with the winning pair.
     */
    pub(super) fn get_tokens_to_update(
        &self,
        winning_pair: (u32, u32),
    ) -> Result<HashMap<(u32, u32), HashMap<u32, Vec<u32>>>, TokenizationError> {
        // It holds the occurrences that we need to remove from the ocurrences list of each candidate.
        // Key: The candidate pair that we need to update
        // Value: word id -> byte offsets of the pair in that word that must be removed.
        let mut to_update: HashMap<(u32, u32), HashMap<u32, Vec<u32>>> = HashMap::new();

        let pair_ocurrences = self
            .tokens
            .get(&winning_pair)
            .ok_or_else(|| {
                TokenizationError::MissingToken(
                    String::from_utf8_lossy(&self.pair_token_info(winning_pair).to_bytes())
                        .into_owned(),
                )
            })?;

        // iterating over all words and offsets where the winning pair occurs.
        // this allows us to reconstruct the candidates that need to be updated.
        for (word_id, offsets) in pair_ocurrences.get_ocurrences() {
            let word_tokens = self
                .word_tokens
                .get(*word_id as usize)
                .ok_or_else(|| {
                    TokenizationError::UnexpectedError(format!(
                        "Unexpected missing word id {}.",
                        word_id
                    ))
                })?;
            self.find_and_insert_adyacent_tokens(
                &mut to_update,
                winning_pair,
                *word_id,
                word_tokens,
                offsets,
            )?;
        }
        Ok(to_update)
    }

    /**
     * Given the candidate occurrences to remove, remove those occurrences
     * from the candidates' occurrence lists.
     */
    pub(super) fn update_tokens(
        &mut self,
        to_update: &HashMap<(u32, u32), HashMap<u32, Vec<u32>>>,
    ) -> Result<(), TokenizationError> {
        for (pair, ocurrences_to_remove) in to_update {
            if let Some(pair_ocurrences) = self.tokens.get_mut(pair) {
                // Iterating over the ocurrences of the pair that needs to be updated
                for (word_id, offsets_to_remove) in ocurrences_to_remove {
                    // Removing the requested occurrences. The removal is
                    // offset-set based because @to_update can contain
                    // duplicate offsets, and an ordered two-pointer match
                    // would desynchronize on them.
                    let remove_set: HashSet<u32> = offsets_to_remove.iter().copied().collect();
                    let new_offsets: Vec<u32> = match pair_ocurrences.get_ocurrences().get(word_id)
                    {
                        Some(current_offsets) => current_offsets
                            .iter()
                            .copied()
                            .filter(|offset| !remove_set.contains(offset))
                            .collect(),
                        None => {
                            return Err(TokenizationError::UnexpectedError(format!(
                                "Word id {} not found in the candidate ocurrences.",
                                word_id
                            )));
                        }
                    };
                    let word_freq = self
                        .word_freqs
                        .get(*word_id as usize)
                        .copied()
                        .ok_or_else(|| {
                            TokenizationError::UnexpectedError(format!(
                                "Unexpected missing word id {}.",
                                word_id
                            ))
                        })?;
                    pair_ocurrences.replace_positions(*word_id, word_freq, new_offsets)?;
                }
            }
        }
        Ok(())
    }
}
