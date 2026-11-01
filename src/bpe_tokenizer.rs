
use crate::{Token, TokenInfo, TokenOcurrences, OcurrencePosition, TokenizationError};

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BinaryHeap;
use std::fmt::Debug;
use std::cmp::Reverse;

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

    /**
     * It initializes the structures of the tokenizer.
     * Iterates over all words in the corpus, and build the initial tokens list.
     */
    fn setup(&mut self) {
        for byte in u8::MIN..=u8::MAX {
            let token = vec![byte];
            let token_id = u32::from(byte);
            self.tokens_id.insert(token.clone(), token_id);
            self.id_tokens.insert(token_id, token.clone());
            self.vocabulary.push(Token::new(token_id, token_id, None));
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
    fn insert_new_token(left: &TokenInfo, right: &TokenInfo,
        new_tokens: &mut HashSet<TokenInfo>,
        tokens: &mut HashMap<TokenInfo, TokenOcurrences>,
        word: &String, count: u64,
        position: (OcurrencePosition, Option<OcurrencePosition>)) {
        let candidate_token: TokenInfo = TokenInfo::merge(left, right);
        // inserting the ocurrence in the new tokens map if it is not an existing token.
        if !tokens.contains_key(&candidate_token) {
           new_tokens.insert(candidate_token.clone());
        }

        tokens.entry(candidate_token)
            .or_insert_with(TokenOcurrences::new)
            .add_ocurrence(word.clone(), count, position);
    }

    /**
     * It merges the initial tokens and prepare for full token iterations.
     */
    fn quickoff(&mut self) -> Result<(), TokenizationError> {
        let mut new_tokens: HashSet<TokenInfo> = HashSet::new();
        // Iterate over all word tokens.
        for (word, word_tokens) in self.word_tokens.iter_mut() {
           // Get the number of ocurrences.
           let count = match self.word_ocurrences.get(word) {
               Some(&ocurrences) => {
                   ocurrences
               }
               None => {
                   return Err(TokenizationError::WordNotFound(word.clone()));
               }
           };

           if word_tokens.is_empty() {
             return Err(TokenizationError::UnexpectedError("Empty tokens list in the word {word}".to_string()));
           }
           let mut index: usize = 1;
           let mut byte_index: usize = 0;
           // iterates over all tokens that forms the current word and creates new token candidates by merging adyacent tokens.
           while index < word_tokens.len() {
               let prev: &TokenInfo = &word_tokens[index as usize - 1];
               let current: &TokenInfo = &word_tokens[index as usize];
               let position = (OcurrencePosition::new(byte_index, prev.size()),
                                                     Some(OcurrencePosition::new(byte_index + prev.size(), current.size())));
               Self::insert_new_token(prev, current, &mut new_tokens, &mut self.tokens, word, count, position);

               index += 1;
               byte_index += prev.size();
           }
        }
        // Insert the initial set of tokens into the tokens heap.
        for token_info in new_tokens {
            match self.tokens.get(&token_info) {
                Some(token_ocurrences) => {
                    self.max_heap.push((
                        token_ocurrences.get_ocurrences_count(),
                        Reverse(token_info),
                    ));
                },
                None => {
                    return Err(TokenizationError::MissingToken(token_info.to_string()));
                }
            }
        }
       Ok(()) 
    }

    pub fn initialize(&mut self) -> Result<(), TokenizationError> {
        if self.initialized {
            return Err(TokenizationError::UnexpectedError("Already initialized".to_string()));
        }
        self.setup();
        match self.quickoff() {
            Err(e) => { return Err(e); } 
            Ok(_) => {}
        }
        self.initialized = true;
        Ok(())
    }

    /**
     * Given a token candidate, find which token ids it was formed from.
     * The left and right token must exists in the id registry.
     */
    fn token_parts(&mut self, winning_token: &TokenInfo) -> Result<(u32, u32), TokenizationError> {
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
            return Err(TokenizationError::MissingToken(winning_token.to_string().clone()));
          }
        };
        return Ok((left_id, right_id));
    }

    /**
     * Inserts an ocurrence position into the corresponding ocurrence list for the given word and token.
     */
    fn insert_token_ocurrence(
        map: &mut HashMap<TokenInfo, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>>,
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
    fn find_and_insert_adyacent_tokens(map: &mut HashMap<TokenInfo, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>>, 
                       winning_token: &TokenInfo,
                       word: &String, word_tokens: &Vec<TokenInfo>,
                       positions: &Vec<(OcurrencePosition, Option<OcurrencePosition>)>) -> Result<(), TokenizationError> {
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
                let (_, prev_right) = &positions[positions_index-1];
                match prev_right {
                    Some(tail) => {
                        if left == tail && !prev_skipped {
                            positions_index += 1;
                            prev_skipped = true;
                            continue;
                        }
                    },
                    None => {
                       return Err(TokenizationError::UnexpectedError("Unexpected empty right part".to_string()));
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
                let token = TokenInfo::new(word_tokens[token_index_left-1].to_bytes(), Some(word_tokens[token_index_left].to_bytes()));
                if &token != winning_token {
                    Self::insert_token_ocurrence(
                        map,
                        token,
                        word,
                        (OcurrencePosition::new(left.get_byte_offset() - word_tokens[token_index_left-1].size(), word_tokens[token_index_left-1].size()),
                            Some(left.clone())),
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
                    if token_index_right+1 < word_tokens.len() {
                        let token = TokenInfo::new(word_tokens[token_index_right].to_bytes(), Some(word_tokens[token_index_right+1].to_bytes()));
                        if &token != winning_token {
                            Self::insert_token_ocurrence(
                                map,
                                token,
                                word,
                        (right.clone(), Some(OcurrencePosition::new(right.get_byte_offset() + right.get_token_length(), word_tokens[token_index_right+1].size()))),
                            );
                        }
                    }
                }
                None => {
                   return Err(TokenizationError::UnexpectedError("Unexpected empty right part in token.".to_string()));
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
    fn get_tokens_to_update(&self, winning_token: &TokenInfo) -> Result<HashMap<TokenInfo, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>>, TokenizationError> {
        // It holds the positions that we need to remove from the ocurrences list in a given token for a given word.
        // Key: The token candidate that we need to update
        // Value: a word -> token list indexes map, representing which ocurrences of the token list for the word must be removed.
        let mut to_update: HashMap<TokenInfo, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>> = HashMap::new();

        // Updating all the tokens that overlaps with the winning token, removing it from the count. 
        match self.tokens.get(&winning_token) {
            Some(token_ocurrences) => {
                // iterating over all words and positions where the winning token was formed from.
                // this allows us to reconstruct the tokens that need to be updated.
                for (word, winning_token_positions) in token_ocurrences.get_ocurrences() {
                   match self.word_tokens.get(word) {
                     Some(word_tokens) => {
                        match Self::find_and_insert_adyacent_tokens(&mut to_update, winning_token, word, word_tokens, winning_token_positions) {
                            Err(e)  => { return Err(e) ; }
                            Ok(_) => { () }
                        }
                     },
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
    fn update_tokens(&mut self, to_update: &HashMap<TokenInfo, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>>) -> Result<(), TokenizationError> {
        for (token_to_update, ocurrences_to_remove) in to_update {
            match self.tokens.get_mut(&token_to_update) {
                Some(ocurrences_to_update ) => {
                    // Iterating over the ocurrences of the token that needs to be updated
                    for word in ocurrences_to_remove.keys() {
                        let word_ocurrencies = match self.word_ocurrences.get(word) {
                            Some(wo) => wo.clone(),
                            None => {
                                return Err(TokenizationError::WordNotFound(word.clone()));
                            }
                        };
                        let ocurrences_positions_to_update = match ocurrences_to_update.get_ocurrences().get(word) {
                            Some(op) => { op }
                            None => { return Err(TokenizationError::WordNotFound(word.clone())); } 
                        };

                        // Finding the ocurrence to remove for the current word and token
                        let mut final_positions: Vec<(OcurrencePosition, Option<OcurrencePosition>)> = Vec::new();
                        match ocurrences_to_remove.get(word) {
                            Some(positions_to_remove) => {
                                let mut rem_index: usize = 0;
                                for ocurrence_position in ocurrences_positions_to_update {
                                    if rem_index < positions_to_remove.len() && positions_to_remove[rem_index] == *ocurrence_position {
                                      // removing this one.
                                      rem_index+=1;
                                      continue;
                                    }
                                    final_positions.push(ocurrence_position.clone());
                                }
                                match ocurrences_to_update.replace_positions(word.clone(), word_ocurrencies, final_positions) {
                                    Result::Ok(_) => (),
                                    Err(err) => { return Err(err); }
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
                            if tm.get_ocurrences_count() >  0 {
                               self.max_heap.push((tm.get_ocurrences_count(), Reverse(token_info)));
                            } else {
                               self.tokens.remove(&token_info);
                            }
                        },
                        None => {
                            // The heap can contain stale entries after a token was removed.
                            continue;
                        }
                    }
                },
                None => return Err(TokenizationError::UnexpectedError("Unexpected empty heap.".to_string())),
            }
        }
        Ok(None)
    }

    fn merge_winning_token_and_generate_new_ones(&mut self, winning_token: &TokenInfo) -> Result<(), TokenizationError> {
        // Update the word tokens and generate new token from winning token adyacent ones.
        let mut new_tokens: HashSet<TokenInfo> = HashSet::new();
        let mut pending_tokens = Vec::new();
        match self.tokens.get(&winning_token) {
           Some(winning_token_ocurrences) => {

              // iterating over all words in which the winning token is present.
              for (word, wining_token_ocurrences_positions) in winning_token_ocurrences.get_ocurrences() {
                 let word_ocurrences = match self.word_ocurrences.get(word) {
                    Some(wo) => { wo.clone() },
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
                          while ocurrences_id < wining_token_ocurrences_positions.len() && 
                             &byte_count > wining_token_ocurrences_positions[ocurrences_id].0.get_byte_offset() {
                             ocurrences_id += 1;
                          }

                          // at this point we know that we can merge tokens at index and index+1
                          if ocurrences_id < wining_token_ocurrences_positions.len() &&
                             &byte_count == wining_token_ocurrences_positions[ocurrences_id].0.get_byte_offset() {
                            // merging winning tokens part into one token
                            let new_token = TokenInfo::new(tokens[index].to_bytes(), Some(tokens[index+1].to_bytes()));
                            if final_tokens.len() > 0 {
                               // if there is a previous token, we merge it with the winning token, resulting in a branch new token.
                               let left = final_tokens[final_tokens.len()-1].clone();
                               // position where this new token appears.
                               let position = (OcurrencePosition::new(byte_count - left.size(), left.size()),
                                       Some(OcurrencePosition::new(byte_count, new_token.size())));
                               pending_tokens.push((left, new_token.clone(), word.clone(), word_ocurrences, position));
                            }
                            final_tokens.push(new_token);
                            prev_new_token = true;

                            // we advance 1 extra position here so next iteration falls at a new ocurrence, not at the second part of this ocurrence.
                            byte_count += tokens[index].size();
                            index += 1;
                          } else {
                            // This means the ocurrences of the tokens to merge is to the right of the current token index
                            if final_tokens.len() > 0 && prev_new_token {
                                let left = final_tokens[final_tokens.len()-1].clone();
                                let position = (OcurrencePosition::new(byte_count - left.size(), left.size()),
                                       Some(OcurrencePosition::new(byte_count, tokens[index].size())));
                                pending_tokens.push((left, tokens[index].clone(), word.clone(), word_ocurrences, position));       
                            }
                            final_tokens.push(tokens[index].clone());
                            prev_new_token = false;
                          }
                          byte_count += tokens[index].size();
                          index += 1;
                       }
                       self.word_tokens.insert(word.clone(), final_tokens);
                    },
                    None => {
                        return Err(TokenizationError::UnexpectedError("Unexpected missing word".to_string()));
                    }
                 }
              }
           },
           None => {
              return Err(TokenizationError::UnexpectedError("Unexpected non existing token.".to_string()));
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
                    self.max_heap.push((
                        token_ocurrences.get_ocurrences_count(),
                        Reverse(token_info),
                    ));
                },
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
            Ok(wt) => { wt },
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
           Ok((left_id, right_id)) => {
             (left_id, right_id)
           },
           Err(e) => { 
            return Err(e); 
           },
        };

        // Adding the new token to the vocabulary and the token id maps.
        self.vocabulary.push(Token::new(token_id, left_id, Some(right_id)));
        self.tokens_id.insert(winning_token.to_bytes(), token_id);
        self.id_tokens.insert(token_id, winning_token.to_bytes());

        // It holds the positions that we need to remove from the ocurrences list in a given token for a a given word.
        // Key: The token candidate that we need to update
        // Value: a word -> token list indexes map, representing which ocurrences in the token list for the word must be removed.
        let to_update: HashMap<TokenInfo, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>> = match self.get_tokens_to_update(&winning_token) {
            Ok(tu) => { tu }
            Err(e) => { return Err(e); }
        };
        // Update the overlapping tokens, removing the ocurrences that overlaps with the winning token.
        match self.update_tokens(&to_update) {
            Ok(_) => (),
            Err(e) => { return Err(e); }
        }

        match self.merge_winning_token_and_generate_new_ones(&winning_token) {
            Ok(_) => {},
            Err(e) => { return Err(e); }
        }

        Ok(Some((token_id, winning_token)))
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenizer_initial_setup() {
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("hello".to_string(), 1);
        word_ocurrences.insert("world".to_string(), 1);

        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.setup();

        assert_eq!(tokenizer.word_ocurrences.len(), 2);
        assert_eq!(tokenizer.tokens.len(), 7); // Each character is a token
        assert_eq!(tokenizer.word_tokens.len(), 2); // Two words: "hello" and "world"
        assert_eq!(tokenizer.max_heap.len(), 0); // No new tokens yet
        assert_eq!(tokenizer.tokens_id.len(), 256);
        assert_eq!(tokenizer.id_tokens.len(), 256);
        assert_eq!(tokenizer.vocabulary.len(), 256);
        for byte in u8::MIN..=u8::MAX {
            assert_eq!(tokenizer.tokens_id.get(&vec![byte]), Some(&u32::from(byte)));
        }

        assert_eq!(tokenizer.word_tokens.get("hello").unwrap().len(), 5); // "h", "e", "l", "l", "o"
        assert_eq!(tokenizer.word_tokens.get("world").unwrap().len(), 5); // "w", "o", "r", "l", "d"
        
        let token_ocurrences_cases : Vec<(&str, u64)> = vec![
            ("h", 1),
            ("e", 1),
            ("l", 3), // "l" appears three times in "hello"
            ("o", 2), // "o" appears in both "hello" and "world"
            ("w", 1),
            ("r", 1),
            ("d", 1),
        ];

        for (token, expected_count) in token_ocurrences_cases {
            let temp_token_info = TokenInfo::new(token.as_bytes().to_vec(), None);
            assert!(tokenizer.tokens.contains_key(&temp_token_info));
            assert_eq!(tokenizer.tokens.get(&temp_token_info).unwrap().get_ocurrences_count(), expected_count);
        }
 
        let token_position_cases: Vec<
            (&str, HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>),
        > = vec![
            (
                "h",
                HashMap::from([(
                    "hello".to_string(),
                    vec![(OcurrencePosition::new(0, 1), None)],
                )]),
            ),
            (
                "e",
                HashMap::from([(
                    "hello".to_string(),
                    vec![(OcurrencePosition::new(1, 1), None)],
                )]),
            ),
            (
                "l",
                HashMap::from([
                    (
                        "hello".to_string(),
                        vec![
                            (OcurrencePosition::new(2, 1), None),
                            (OcurrencePosition::new(3, 1), None),
                        ],
                    ),
                    (
                        "world".to_string(),
                        vec![(OcurrencePosition::new(3, 1), None)],
                    ),
                ]),
            ),
            (
                "o",
                HashMap::from([
                    (
                        "hello".to_string(),
                        vec![(OcurrencePosition::new(4, 1), None)],
                    ),
                    (
                        "world".to_string(),
                        vec![(OcurrencePosition::new(1, 1), None)],
                    ),
                ]),
            ),
            (
                "w",
                HashMap::from([(
                    "world".to_string(),
                    vec![(OcurrencePosition::new(0, 1), None)],
                )]),
            ),
            (
                "r",
                HashMap::from([(
                    "world".to_string(),
                    vec![(OcurrencePosition::new(2, 1), None)],
                )]),
            ),
            (
                "d",
                HashMap::from([(
                    "world".to_string(),
                    vec![(OcurrencePosition::new(4, 1), None)],
                )]),
            ),
        ];

        for (token, expected_positions) in token_position_cases {
            let temp_token_info = TokenInfo::new(token.as_bytes().to_vec(), None);
            assert!(tokenizer.tokens.contains_key(&temp_token_info));
            let actual_positions = tokenizer.tokens.get(&temp_token_info).unwrap().get_ocurrences();
            assert_eq!(actual_positions, &expected_positions);
        }
    }

    #[test]
    fn test_bpe_tokenizer_quickoff() {
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("hello".to_string(), 1);
        word_ocurrences.insert(" worlod".to_string(), 1);

        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.setup();
        let result: Result<(), TokenizationError> = tokenizer.quickoff();

        assert!(result.is_ok());

        let hello_generated_tokens_count = 4;
        let worlod_generated_tokens_count = 6;
        let common_tokens_count = 1; // "lo" is common tokens generated from both words
        println!("tokens_count: {}", tokenizer.tokens.len());
        let expected_tokens_count = (8 + hello_generated_tokens_count + worlod_generated_tokens_count) - common_tokens_count;
        assert!(tokenizer.tokens.len() == (expected_tokens_count as usize));

        // Check that new tokens have been added to the max_heap
        assert!(!tokenizer.max_heap.is_empty());

        // Check that the new tokens are present in the tokens map
        for (count, Reverse(token_info)) in &tokenizer.max_heap {
            assert!(tokenizer.tokens.contains_key(token_info));
            assert_eq!(tokenizer.tokens.get(token_info).unwrap().get_ocurrences_count(), *count);
        }
        let most_frequent = tokenizer.max_heap.pop();
        assert!(most_frequent.is_some());
        let (count, Reverse(token_info)) = most_frequent.unwrap();
        assert!(count == 2); // In this simple case, all new tokens should have a count of 2
        assert!(token_info.to_string() == "lo");

        let second_most_frequent = tokenizer.max_heap.pop();
        let (count1, Reverse(token_info1)) = second_most_frequent.unwrap();
        assert!(count1 == 1);
        assert!(token_info1.to_string() == " w");
    }

    #[test]
    fn setup_splits_unicode_words_into_utf8_bytes() {
        let word = "é".to_string();
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert(word.clone(), 1);

        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.setup();

        let byte_tokens = tokenizer.word_tokens.get(&word).unwrap();
        assert_eq!(byte_tokens.len(), 2);
        assert_eq!(byte_tokens[0].to_bytes(), [0xC3]);
        assert_eq!(byte_tokens[1].to_bytes(), [0xA9]);
        assert_eq!(tokenizer.tokens_id.get(&vec![0xC3]), Some(&195));
        assert_eq!(tokenizer.tokens_id.get(&vec![0xA9]), Some(&169));
        assert_eq!(
            tokenizer.tokens[&TokenInfo::new(vec![0xC3], None)]
                .get_ocurrences()["é"][0]
                .0
                .get_byte_offset(),
            &0
        );
        assert_eq!(
            tokenizer.tokens[&TokenInfo::new(vec![0xA9], None)]
                .get_ocurrences()["é"][0]
                .0
                .get_byte_offset(),
            &1
        );
    }

    #[test]
    fn setup_with_empty_corpus_initializes_only_the_byte_vocabulary() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());

        tokenizer.setup();

        assert_eq!(tokenizer.vocabulary.len(), 256);
        assert_eq!(tokenizer.tokens_id.len(), 256);
        assert!(tokenizer.tokens.is_empty());
        assert!(tokenizer.word_tokens.is_empty());
    }

    #[test]
    fn setup_ignores_zero_frequency_words() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 0)]));

        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        assert!(tokenizer.word_tokens.is_empty());
        assert!(tokenizer.tokens.is_empty());
        assert!(tokenizer.max_heap.is_empty());
    }

    #[test]
    fn learned_token_ids_start_after_the_256_byte_tokens() {
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("ab".to_string(), 1);
        let mut tokenizer = BPETokenizer::new(word_ocurrences);

        tokenizer.initialize().unwrap();
        let (token_id, token) = tokenizer.next_token().unwrap().unwrap();

        assert_eq!(token_id, 256);
        assert_eq!(token.to_bytes(), b"ab");
        assert_eq!(tokenizer.tokens_id.get(b"ab".as_slice()), Some(&256));
    }

    #[test]
    fn insert_new_token_tracks_occurrences_and_deduplicates_candidates() {
        let mut tokens = HashMap::new();
        let mut new_tokens = HashSet::new();
        let left = TokenInfo::new(b"a".to_vec(), None);
        let right = TokenInfo::new(b"b".to_vec(), None);
        let merged = TokenInfo::merge(&left, &right);
        let position = (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1)));

        BPETokenizer::insert_new_token(
            &left,
            &right,
            &mut new_tokens,
            &mut tokens,
            &"ab".to_string(),
            3,
            position.clone(),
        );
        BPETokenizer::insert_new_token(
            &left,
            &right,
            &mut new_tokens,
            &mut tokens,
            &"ab".to_string(),
            3,
            position.clone(),
        );

        assert_eq!(new_tokens.len(), 1);
        assert!(new_tokens.contains(&merged));
        assert_eq!(tokens[&merged].get_ocurrences_count(), 6);
        assert_eq!(tokens[&merged].get_ocurrences()["ab"].len(), 2);
        assert_eq!(tokens[&merged].get_ocurrences()["ab"][0], position);
        assert_eq!(tokens[&merged].get_ocurrences()["ab"][1], position);
    }

    #[test]
    fn insert_new_token_does_not_mark_an_existing_candidate_as_new() {
        let left = TokenInfo::new(b"a".to_vec(), None);
        let right = TokenInfo::new(b"b".to_vec(), None);
        let merged = TokenInfo::merge(&left, &right);
        let position = (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1)));
        let mut tokens = HashMap::new();
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence("ab".to_string(), 2, position.clone());
        tokens.insert(merged.clone(), occurrences);
        let mut new_tokens = HashSet::new();

        BPETokenizer::insert_new_token(
            &left,
            &right,
            &mut new_tokens,
            &mut tokens,
            &"ab".to_string(),
            3,
            position,
        );

        assert!(new_tokens.is_empty());
        assert_eq!(tokens[&merged].get_ocurrences_count(), 5);
    }

    #[test]
    fn token_parts_resolves_left_and_right_byte_fragments() {
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("ab".to_string(), 1);
        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();

        let merged = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        assert_eq!(tokenizer.token_parts(&merged).unwrap(), (97, 98));
    }

    #[test]
    fn token_parts_rejects_a_token_without_a_right_part() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        tokenizer.tokens_id.insert(b"a".to_vec(), 97);

        assert!(matches!(
            tokenizer.token_parts(&TokenInfo::new(b"a".to_vec(), None)),
            Err(TokenizationError::MissingToken(_))
        ));
    }

    #[test]
    fn update_tokens_errors_when_word_is_not_in_candidate_occurrences() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([
            ("ab".to_string(), 1),
            ("missing".to_string(), 1),
        ]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();
        let token = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let to_remove = HashMap::from([(
            token,
            HashMap::from([(
                "missing".to_string(),
                vec![(OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1)))],
            )]),
        )]);

        assert!(matches!(
            tokenizer.update_tokens(&to_remove),
            Err(TokenizationError::WordNotFound(word)) if word == "missing"
        ));
    }

    #[test]
    fn quickoff_succeeds_for_empty_corpus() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());

        assert!(tokenizer.quickoff().is_ok());
        assert!(tokenizer.max_heap.is_empty());
    }

    #[test]
    fn merge_winning_token_and_generate_new_ones_updates_word_tokens() {
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("ab".to_string(), 1);
        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();

        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        tokenizer.merge_winning_token_and_generate_new_ones(&winning).unwrap();

        assert_eq!(tokenizer.word_tokens["ab"], vec![winning]);
    }

    #[test]
    fn insert_token_ocurrence_appends_positions_for_a_word() {
        let mut map = HashMap::new();
        let token = TokenInfo::new(b"ab".to_vec(), None);
        let pos = (OcurrencePosition::new(0, 2), None);

        BPETokenizer::insert_token_ocurrence(&mut map, token.clone(), "ab", pos.clone());
        BPETokenizer::insert_token_ocurrence(&mut map, token.clone(), "ba", pos.clone());

        assert_eq!(map[&token]["ab"], vec![pos.clone()]);
        assert_eq!(map[&token]["ba"], vec![pos]);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_registers_left_overlap() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"b".to_vec(), Some(b"c".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"b".to_vec(), None),
            TokenInfo::new(b"c".to_vec(), None),
        ];
        let positions = vec![(OcurrencePosition::new(1, 1), Some(OcurrencePosition::new(2, 1)))];

        BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"abc".to_string(),
            &word_tokens,
            &positions,
        )
        .unwrap();

        let adjacent = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        assert_eq!(
            map[&adjacent]["abc"],
            vec![(OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1)))]
        );
    }

    #[test]
    fn find_and_insert_adyacent_tokens_registers_right_overlap() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"b".to_vec(), Some(b"c".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"b".to_vec(), None),
            TokenInfo::new(b"c".to_vec(), None),
            TokenInfo::new(b"d".to_vec(), None),
        ];
        let positions = vec![(OcurrencePosition::new(1, 1), Some(OcurrencePosition::new(2, 1)))];

        BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"abcd".to_string(),
            &word_tokens,
            &positions,
        )
        .unwrap();

        let left_adjacent = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let adjacent = TokenInfo::new(b"c".to_vec(), Some(b"d".to_vec()));
        assert_eq!(
            map[&left_adjacent]["abcd"],
            vec![(OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1)))]
        );
        assert_eq!(
            map[&adjacent]["abcd"],
            vec![(OcurrencePosition::new(2, 1), Some(OcurrencePosition::new(3, 1)))]
        );
    }

    #[test]
    fn find_and_insert_adyacent_tokens_uses_byte_offsets_for_multibyte_tokens() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"b".to_vec(), Some(b"cc".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"aa".to_vec(), None),
            TokenInfo::new(b"b".to_vec(), None),
            TokenInfo::new(b"cc".to_vec(), None),
            TokenInfo::new(b"d".to_vec(), None),
        ];
        let positions = vec![(OcurrencePosition::new(2, 1), Some(OcurrencePosition::new(3, 2)))];

        BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"aabccd".to_string(),
            &word_tokens,
            &positions,
        )
        .unwrap();

        let left_adjacent = TokenInfo::new(b"aa".to_vec(), Some(b"b".to_vec()));
        let right_adjacent = TokenInfo::new(b"cc".to_vec(), Some(b"d".to_vec()));
        assert_eq!(
            map[&left_adjacent]["aabccd"],
            vec![(OcurrencePosition::new(0, 2), Some(OcurrencePosition::new(2, 1)))]
        );
        assert_eq!(
            map[&right_adjacent]["aabccd"],
            vec![(OcurrencePosition::new(3, 2), Some(OcurrencePosition::new(5, 1)))]
        );
    }

    #[test]
    fn find_and_insert_adyacent_tokens_handles_winner_at_word_start() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"b".to_vec(), None),
            TokenInfo::new(b"c".to_vec(), None),
        ];
        let positions = vec![(OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1)))];

        BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"abc".to_string(),
            &word_tokens,
            &positions,
        )
        .unwrap();

        let adjacent = TokenInfo::new(b"b".to_vec(), Some(b"c".to_vec()));
        assert_eq!(
            map[&adjacent]["abc"],
            vec![(OcurrencePosition::new(1, 1), Some(OcurrencePosition::new(2, 1)))]
        );
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_handles_winner_at_word_end() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"b".to_vec(), Some(b"c".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"b".to_vec(), None),
            TokenInfo::new(b"c".to_vec(), None),
        ];
        let positions = vec![(OcurrencePosition::new(1, 1), Some(OcurrencePosition::new(2, 1)))];

        BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"abc".to_string(),
            &word_tokens,
            &positions,
        )
        .unwrap();

        let adjacent = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        assert_eq!(
            map[&adjacent]["abc"],
            vec![(OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1)))]
        );
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_skips_overlapping_winner_positions() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"a".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"a".to_vec(), None),
        ];
        let positions = vec![
            (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
            (OcurrencePosition::new(1, 1), Some(OcurrencePosition::new(2, 1))),
        ];

        BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"aaa".to_string(),
            &word_tokens,
            &positions,
        )
        .unwrap();

        assert!(map.is_empty());
    }

    #[test]
    fn find_and_insert_adyacent_tokens_accepts_no_occurrences() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"b".to_vec(), None),
        ];

        BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"ab".to_string(),
            &word_tokens,
            &Vec::new(),
        )
        .unwrap();

        assert!(map.is_empty());
    }

    #[test]
    fn find_and_insert_adyacent_tokens_rejects_occurrence_without_right_part() {
        let mut map = HashMap::new();
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let word_tokens = vec![
            TokenInfo::new(b"a".to_vec(), None),
            TokenInfo::new(b"b".to_vec(), None),
        ];
        let positions = vec![(OcurrencePosition::new(0, 1), None)];

        assert!(BPETokenizer::find_and_insert_adyacent_tokens(
            &mut map,
            &winning,
            &"ab".to_string(),
            &word_tokens,
            &positions,
        )
        .is_err());
    }

    #[test]
    fn get_tokens_to_update_tracks_overlap_with_winning_token() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abc".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = TokenInfo::new(b"b".to_vec(), Some(b"c".to_vec()));
        let updates = tokenizer.get_tokens_to_update(&winning).unwrap();
        let adjacent = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));

        assert!(updates.contains_key(&adjacent));
        assert_eq!(updates[&adjacent]["abc"].len(), 1);
    }

    #[test]
    fn get_tokens_to_update_accumulates_repeated_adjacent_candidates() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("xabyxaby".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let updates = tokenizer.get_tokens_to_update(&winning).unwrap();
        let left_adjacent = TokenInfo::new(b"x".to_vec(), Some(b"a".to_vec()));
        let right_adjacent = TokenInfo::new(b"b".to_vec(), Some(b"y".to_vec()));

        assert_eq!(updates[&left_adjacent]["xabyxaby"].len(), 2);
        assert_eq!(updates[&right_adjacent]["xabyxaby"].len(), 2);
        assert_eq!(
            updates[&left_adjacent]["xabyxaby"],
            vec![
                (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
                (OcurrencePosition::new(4, 1), Some(OcurrencePosition::new(5, 1))),
            ]
        );
        assert_eq!(
            updates[&right_adjacent]["xabyxaby"],
            vec![
                (OcurrencePosition::new(2, 1), Some(OcurrencePosition::new(3, 1))),
                (OcurrencePosition::new(6, 1), Some(OcurrencePosition::new(7, 1))),
            ]
        );
    }

    #[test]
    fn get_tokens_to_update_returns_empty_when_winner_has_no_neighbors() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        assert!(tokenizer.get_tokens_to_update(&winning).unwrap().is_empty());
    }

    #[test]
    fn get_tokens_to_update_errors_for_missing_winner() {
        let tokenizer = BPETokenizer::new(HashMap::new());
        let missing = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));

        assert!(matches!(
            tokenizer.get_tokens_to_update(&missing),
            Err(TokenizationError::MissingToken(_))
        ));
    }

    #[test]
    fn get_tokens_to_update_errors_when_word_tokens_are_missing() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abc".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();
        tokenizer.word_tokens.remove("abc");
        let winning = TokenInfo::new(b"b".to_vec(), Some(b"c".to_vec()));

        assert!(matches!(
            tokenizer.get_tokens_to_update(&winning),
            Err(TokenizationError::WordNotFound(word)) if word == "abc"
        ));
    }

    #[test]
    fn update_tokens_removes_overlapping_positions() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abc".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = TokenInfo::new(b"b".to_vec(), Some(b"c".to_vec()));
        let updates = tokenizer.get_tokens_to_update(&winning).unwrap();
        tokenizer.update_tokens(&updates).unwrap();

        let adjacent = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        assert_eq!(tokenizer.tokens[&adjacent].get_ocurrences_count(), 0);
        assert!(!tokenizer.tokens[&adjacent].get_ocurrences().contains_key("abc"));
    }

    #[test]
    fn update_tokens_preserves_positions_not_requested_for_removal() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abab".to_string(), 2)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();
        let adjacent = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let original = tokenizer.tokens[&adjacent].get_ocurrences()["abab"].clone();
        assert_eq!(tokenizer.tokens[&adjacent].get_ocurrences_count(), 4);
        let to_remove = HashMap::from([(
            adjacent.clone(),
            HashMap::from([("abab".to_string(), vec![original[0].clone()])]),
        )]);

        tokenizer.update_tokens(&to_remove).unwrap();

        assert_eq!(
            tokenizer.tokens[&adjacent].get_ocurrences()["abab"],
            vec![original[1].clone()]
        );
        assert_eq!(tokenizer.tokens[&adjacent].get_ocurrences_count(), 2);
    }

    #[test]
    fn update_tokens_errors_when_word_frequency_is_missing() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let token = TokenInfo::new(b"ab".to_vec(), None);
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(
            "ab".to_string(),
            1,
            (OcurrencePosition::new(0, 2), None),
        );
        tokenizer.tokens.insert(token.clone(), occurrences);
        let to_remove = HashMap::from([(
            token,
            HashMap::from([(
                "ab".to_string(),
                vec![(OcurrencePosition::new(0, 2), None)],
            )]),
        )]);

        assert!(matches!(
            tokenizer.update_tokens(&to_remove),
            Err(TokenizationError::WordNotFound(word)) if word == "ab"
        ));
    }

    #[test]
    fn update_tokens_ignores_a_candidate_absent_from_token_map() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let absent = TokenInfo::new(b"ab".to_vec(), None);
        let to_remove = HashMap::from([(
            absent,
            HashMap::from([(
                "ab".to_string(),
                vec![(OcurrencePosition::new(0, 2), None)],
            )]),
        )]);

        assert!(tokenizer.update_tokens(&to_remove).is_ok());
    }

    #[test]
    fn update_tokens_errors_when_requested_word_has_no_occurrence_entry() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();
        let token = TokenInfo::new(b"a".to_vec(), None);
        let to_remove = HashMap::from([(
            token,
            HashMap::from([(
                "missing".to_string(),
                vec![(OcurrencePosition::new(0, 1), None)],
            )]),
        )]);

        assert!(matches!(
            tokenizer.update_tokens(&to_remove),
            Err(TokenizationError::WordNotFound(word)) if word == "missing"
        ));
    }

    #[test]
    fn get_winning_token_skips_stale_heap_entries() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let stale = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let valid = TokenInfo::new(b"c".to_vec(), Some(b"d".to_vec()));

        tokenizer.tokens.insert(stale.clone(), TokenOcurrences::new());

        let mut valid_ocurrences = TokenOcurrences::new();
        valid_ocurrences.add_ocurrence(
            "cd".to_string(),
            5,
            (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
        );
        tokenizer.tokens.insert(valid.clone(), valid_ocurrences);

        tokenizer.max_heap.push((10, Reverse(stale.clone())));
        tokenizer.max_heap.push((5, Reverse(valid.clone())));

        assert_eq!(tokenizer.get_winning_token().unwrap(), Some(valid));
        assert!(!tokenizer.tokens.contains_key(&stale));
    }

    #[test]
    fn get_winning_token_refreshes_a_stale_positive_count() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let token = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(
            "ab".to_string(),
            3,
            (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
        );
        tokenizer.tokens.insert(token.clone(), occurrences);
        tokenizer.max_heap.push((10, Reverse(token.clone())));

        assert_eq!(tokenizer.get_winning_token().unwrap(), Some(token.clone()));
        assert!(tokenizer.max_heap.is_empty());
    }

    #[test]
    fn get_winning_token_breaks_equal_counts_by_lexicographically_lowest_token() {
        let mut tokenizer =
            BPETokenizer::new(HashMap::from([("ab".to_string(), 1), ("ba".to_string(), 1)]));
        tokenizer.initialize().unwrap();

        assert_eq!(
            tokenizer.get_winning_token().unwrap(),
            Some(TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec())))
        );
    }

    #[test]
    fn get_winning_token_skips_heap_entry_without_token_record() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let missing = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let valid = TokenInfo::new(b"c".to_vec(), Some(b"d".to_vec()));
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(
            "cd".to_string(),
            1,
            (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
        );
        tokenizer.tokens.insert(valid.clone(), occurrences);
        tokenizer.max_heap.push((2, Reverse(missing)));
        tokenizer.max_heap.push((1, Reverse(valid.clone())));

        assert_eq!(tokenizer.get_winning_token().unwrap(), Some(valid));
    }

    #[test]
    fn initialize_rejects_second_call() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));

        assert!(tokenizer.initialize().is_ok());
        assert!(tokenizer.initialize().is_err());
    }

    #[test]
    fn initialize_does_not_mark_tokenizer_initialized_when_setup_fails() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("".to_string(), 1)]));

        assert!(tokenizer.initialize().is_err());
        assert!(!tokenizer.initialized);
    }

    #[test]
    fn quickoff_errors_when_word_has_no_byte_tokens() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("".to_string(), 1)]));
        tokenizer.setup();

        assert!(tokenizer.quickoff().is_err());
    }

    #[test]
    fn quickoff_errors_when_word_frequency_is_missing() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        tokenizer.word_tokens.insert(
            "ab".to_string(),
            vec![
                TokenInfo::new(b"a".to_vec(), None),
                TokenInfo::new(b"b".to_vec(), None),
            ],
        );

        assert!(matches!(
            tokenizer.quickoff(),
            Err(TokenizationError::WordNotFound(word)) if word == "ab"
        ));
    }

    #[test]
    fn token_parts_errors_for_missing_left_or_right_fragments() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        tokenizer.tokens_id.insert(b"a".to_vec(), 97);
        tokenizer.tokens_id.insert(b"b".to_vec(), 98);

        assert!(tokenizer.token_parts(&TokenInfo::new(b"missing".to_vec(), Some(b"b".to_vec()))).is_err());
        assert!(tokenizer.token_parts(&TokenInfo::new(b"a".to_vec(), Some(b"missing".to_vec()))).is_err());
    }

    #[test]
    fn merge_winning_token_errors_when_candidate_is_missing() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let missing = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));

        assert!(matches!(
            tokenizer.merge_winning_token_and_generate_new_ones(&missing),
            Err(TokenizationError::UnexpectedError(_))
        ));
    }

    #[test]
    fn merge_winning_token_generates_candidates_on_both_sides() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("xaby".to_string(), 2)]));
        tokenizer.initialize().unwrap();
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));

        tokenizer
            .merge_winning_token_and_generate_new_ones(&winning)
            .unwrap();

        assert_eq!(
            tokenizer.word_tokens["xaby"],
            vec![
                TokenInfo::new(b"x".to_vec(), None),
                winning.clone(),
                TokenInfo::new(b"y".to_vec(), None),
            ]
        );
        let left_candidate = TokenInfo::new(b"x".to_vec(), Some(b"ab".to_vec()));
        let right_candidate = TokenInfo::new(b"ab".to_vec(), Some(b"y".to_vec()));
        assert_eq!(tokenizer.tokens[&left_candidate].get_ocurrences_count(), 2);
        assert_eq!(tokenizer.tokens[&right_candidate].get_ocurrences_count(), 2);
        assert!(tokenizer.max_heap.iter().any(|(_, Reverse(token))| token == &left_candidate));
        assert!(tokenizer.max_heap.iter().any(|(_, Reverse(token))| token == &right_candidate));
    }

    #[test]
    fn merge_winning_token_errors_when_word_frequency_is_missing() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(
            "ab".to_string(),
            1,
            (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
        );
        tokenizer.tokens.insert(winning.clone(), occurrences);
        tokenizer.word_tokens.insert(
            "ab".to_string(),
            vec![
                TokenInfo::new(b"a".to_vec(), None),
                TokenInfo::new(b"b".to_vec(), None),
            ],
        );

        assert!(matches!(
            tokenizer.merge_winning_token_and_generate_new_ones(&winning),
            Err(TokenizationError::WordNotFound(word)) if word == "ab"
        ));
    }

    #[test]
    fn merge_winning_token_errors_when_word_tokens_are_missing() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(
            "ab".to_string(),
            1,
            (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
        );
        tokenizer.tokens.insert(winning.clone(), occurrences);

        assert!(matches!(
            tokenizer.merge_winning_token_and_generate_new_ones(&winning),
            Err(TokenizationError::UnexpectedError(_))
        ));
    }

    #[test]
    fn next_token_errors_when_winning_parts_are_not_registered() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let winning = TokenInfo::new(b"unknown-left".to_vec(), Some(b"unknown-right".to_vec()));
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(
            "ab".to_string(),
            1,
            (OcurrencePosition::new(0, 1), Some(OcurrencePosition::new(1, 1))),
        );
        tokenizer.tokens.insert(winning.clone(), occurrences);
        tokenizer.max_heap.push((1, Reverse(winning)));

        assert!(matches!(
            tokenizer.next_token(),
            Err(TokenizationError::MissingToken(_))
        ));
        assert!(tokenizer.vocabulary.is_empty());
    }

    #[test]
    fn next_token_returns_none_when_queue_is_empty() {
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("a".to_string(), 1);

        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();

        assert_eq!(tokenizer.get_winning_token().unwrap(), None);
        assert_eq!(tokenizer.next_token().unwrap(), None);
    }

    #[test]
    fn repeated_byte_word_chain_learns_expected_tokens() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("aaa".to_string(), 1)]));
        tokenizer.initialize().unwrap();

        let first = tokenizer.next_token().unwrap().unwrap();
        assert_eq!(first.0, 256);
        assert_eq!(first.1.to_bytes(), b"aa");

        let second = tokenizer.next_token().unwrap().unwrap();
        assert_eq!(second.0, 257);
        assert_eq!(second.1.to_bytes(), b"aaa");

        assert_eq!(tokenizer.next_token().unwrap(), None);
    }

    #[test]
    fn four_repeated_bytes_merge_without_overlapping_occurrences() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("aaaa".to_string(), 1)]));
        tokenizer.initialize().unwrap();

        let first = tokenizer.next_token().unwrap().unwrap();
        assert_eq!(first.0, 256);
        assert_eq!(first.1.to_bytes(), b"aa");

        let second = tokenizer.next_token().unwrap().unwrap();
        assert_eq!(second.0, 257);
        assert_eq!(second.1.to_bytes(), b"aaaa");

        assert_eq!(tokenizer.next_token().unwrap(), None);
    }
}  