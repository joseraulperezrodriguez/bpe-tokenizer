use std::collections::HashMap;

use crate::{OcurrencePosition, TokenInfo, TokenizationError};

use super::BPETokenizer;

impl BPETokenizer {
    /**
     * Inserts an ocurrence position into the corresponding ocurrence list for the given word and token.
     */
    pub(super) fn insert_token_ocurrence(
        map: &mut HashMap<
            TokenInfo,
            HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>,
        >,
        token_info: TokenInfo,
        word: &str,
        pos: (OcurrencePosition, Option<OcurrencePosition>),
    ) {
        map.entry(token_info)
            .or_default()
            .entry(word.to_owned())
            .or_default()
            .push(pos);
    }

    /**
     * It find and constructs all tokens that has left or right overlapping with the @positions ocurrences
     * @map, where we store the adyacent positions.
     * @word, the word where we are building new tokens.
     * @word_tokens, the list of tokens for ths given word.
     * @positions, the list where we need to check for adyacent nodes.
     */
    pub(super) fn find_and_insert_adyacent_tokens(
        map: &mut HashMap<
            TokenInfo,
            HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>,
        >,
        winning_token: &TokenInfo,
        word: &String,
        word_tokens: &Vec<TokenInfo>,
        positions: &Vec<(OcurrencePosition, Option<OcurrencePosition>)>,
    ) -> Result<(), TokenizationError> {
        let mut token_index_left = 0;
        let mut token_index_cumul_left = 0;

        let mut token_index_right = 0;
        let mut token_index_cumul_right = 0;

        let mut positions_index = 0;
        let mut prev_skipped = false;

        // We iterate over all positions and form its left and right adyacent tokens.
        while positions_index < positions.len() {
            let (left, right_opt) = &positions[positions_index];

            // If the previous position overlaps with this one, we move one position forward,
            // because the current one won't be merge as a new token, so we don't need to find
            // its adyacent nodes to update.
            if positions_index > 0 {
                let (_, prev_right) = &positions[positions_index - 1];
                match prev_right {
                    Some(tail) => {
                        if left == tail && !prev_skipped {
                            positions_index += 1;
                            prev_skipped = true;
                            continue;
                        }
                    }
                    None => {
                        return Err(TokenizationError::UnexpectedError(
                            "Unexpected empty right part".to_string(),
                        ));
                    }
                }
            }
            prev_skipped = false;

            // Move @token_index_left pointer, until it points to the position where @left starts.
            while left.get_byte_offset() > &token_index_cumul_left {
                token_index_cumul_left += word_tokens[token_index_left].size();
                token_index_left += 1;
            }
            // at this point it is guaranteed that @token_index_left < word_tokens.len(),
            // because if @token_index_left points to the left part, then there needs to be another index that points at the right part
            // because @positions is built from the token list in the end.

            // if there is a token before {left} positions, we merge it.
            if token_index_left > 0 {
                let token = TokenInfo::new(
                    word_tokens[token_index_left - 1].to_bytes(),
                    Some(word_tokens[token_index_left].to_bytes()),
                );
                if &token != winning_token {
                    Self::insert_token_ocurrence(
                        map,
                        token,
                        word,
                        (
                            OcurrencePosition::new(
                                left.get_byte_offset() - word_tokens[token_index_left - 1].size(),
                                word_tokens[token_index_left - 1].size(),
                            ),
                            Some(left.clone()),
                        ),
                    );
                }
            }

            match right_opt {
                Some(right) => {
                    // Move @token_index_left pointer, until it points to the position where @left starts.
                    while right.get_byte_offset() > &token_index_cumul_right {
                        token_index_cumul_right += word_tokens[token_index_right].size();
                        token_index_right += 1;
                    }

                    // only in this case, we confirm there is a token before {right} positions
                    if token_index_right + 1 < word_tokens.len() {
                        let token = TokenInfo::new(
                            word_tokens[token_index_right].to_bytes(),
                            Some(word_tokens[token_index_right + 1].to_bytes()),
                        );
                        if &token != winning_token {
                            Self::insert_token_ocurrence(
                                map,
                                token,
                                word,
                                (
                                    right.clone(),
                                    Some(OcurrencePosition::new(
                                        right.get_byte_offset() + right.get_token_length(),
                                        word_tokens[token_index_right + 1].size(),
                                    )),
                                ),
                            );
                        }
                    }
                }
                None => {
                    return Err(TokenizationError::UnexpectedError(
                        "Unexpected empty right part in token.".to_string(),
                    ));
                }
            }

            positions_index += 1;
        }

        Ok(())
    }

    /**
     * Given a @winning_token, it searches over all words and token ocurrences of it and find the tokens that needs
     * to be removed because overlaps with the winning token.
     */
    pub(super) fn get_tokens_to_update(
        &self,
        winning_token: &TokenInfo,
    ) -> Result<
        HashMap<TokenInfo, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>>,
        TokenizationError,
    > {
        // It holds the positions that we need to remove from the ocurrences list in a given token for a given word.
        // Key: The token candidate that we need to update
        // Value: a word -> token list indexes map, representing which ocurrences of the token list for the word must be removed.
        let mut to_update: HashMap<
            TokenInfo,
            HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>,
        > = HashMap::new();

        // Updating all the tokens that overlaps with the winning token, removing it from the count.
        match self.tokens.get(&winning_token) {
            Some(token_ocurrences) => {
                // iterating over all words and positions where the winning token was formed from.
                // this allows us to reconstruct the tokens that need to be updated.
                for (word, winning_token_positions) in token_ocurrences.get_ocurrences() {
                    match self.word_tokens.get(word) {
                        Some(word_tokens) => {
                            match Self::find_and_insert_adyacent_tokens(
                                &mut to_update,
                                winning_token,
                                word,
                                word_tokens,
                                winning_token_positions,
                            ) {
                                Err(e) => {
                                    return Err(e);
                                }
                                Ok(_) => (),
                            }
                        }
                        None => {
                            return Err(TokenizationError::WordNotFound(word.clone()));
                        }
                    }
                }
            }
            None => {
                return Err(TokenizationError::MissingToken(winning_token.to_string()));
            }
        }
        return Ok(to_update);
    }

    /**
     * Given a list of token ocurrences for a given token and word, remove those ocurrences.
     */
    pub(super) fn update_tokens(
        &mut self,
        to_update: &HashMap<
            TokenInfo,
            HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>,
        >,
    ) -> Result<(), TokenizationError> {
        for (token_to_update, ocurrences_to_remove) in to_update {
            match self.tokens.get_mut(&token_to_update) {
                Some(ocurrences_to_update) => {
                    // Iterating over the ocurrences of the token that needs to be updated
                    for word in ocurrences_to_remove.keys() {
                        let word_ocurrencies = match self.word_ocurrences.get(word) {
                            Some(wo) => wo.clone(),
                            None => {
                                return Err(TokenizationError::WordNotFound(word.clone()));
                            }
                        };
                        let ocurrences_positions_to_update =
                            match ocurrences_to_update.get_ocurrences().get(word) {
                                Some(op) => op,
                                None => {
                                    return Err(TokenizationError::WordNotFound(word.clone()));
                                }
                            };

                        // Finding the ocurrence to remove for the current word and token
                        let mut final_positions: Vec<(
                            OcurrencePosition,
                            Option<OcurrencePosition>,
                        )> = Vec::new();
                        match ocurrences_to_remove.get(word) {
                            Some(positions_to_remove) => {
                                let mut rem_index: usize = 0;
                                for ocurrence_position in ocurrences_positions_to_update {
                                    if rem_index < positions_to_remove.len()
                                        && positions_to_remove[rem_index] == *ocurrence_position
                                    {
                                        // removing this one.
                                        rem_index += 1;
                                        continue;
                                    }
                                    final_positions.push(ocurrence_position.clone());
                                }
                                match ocurrences_to_update.replace_positions(
                                    word.clone(),
                                    word_ocurrencies,
                                    final_positions,
                                ) {
                                    Result::Ok(_) => (),
                                    Err(err) => {
                                        return Err(err);
                                    }
                                }
                            }
                            None => {
                                return Err(TokenizationError::WordNotFound(word.clone()));
                            }
                        }
                    }
                }
                None => {}
            }
        }
        Ok(())
    }
}
