
use crate::common::*;

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BinaryHeap;
use std::fmt::Debug;

#[derive(Clone, Debug, Default)]
pub struct BPETokenizer {
    // A map of word occurrences, where the key is the word and the value is the number of occurrences.
    word_ocurrences: HashMap<String, u32>,
    // A set of initial tokens that are considered valid for the tokenizer.
    initial_tokens: HashSet<String>,
    // A map of tokens, where key is a tuple of the left and right tokens, and the value is a TokenOcurrences struct that contains information about the occurrences of the token.
    tokens: HashMap<TokenInfo, TokenOcurrences>,
    // A map of word tokens, where the key is the word and the value is a vector of tuples containing the left token and an optional right token.
    word_tokens: HashMap<String, Vec<TokenInfo>>,
    // A max-heap that stores tuples of (count, left_token, right_token) for efficient retrieval of the most frequent tokens.
    max_heap: BinaryHeap<(u32, TokenInfo)>,
    // A map of tokens to their corresponding IDs.
    tokens_id: HashMap<String, u32>,
    // A map of IDs to their corresponding tokens.
    id_tokens: HashMap<u32, String>,
    // A map of vocabulary, where the key is the token ID and the value is a tuple of (left token ID, optional right token ID).
    vocabulary: HashMap<u32, (u32, Option<u32>)>,
}

impl BPETokenizer {
    pub fn new(word_ocurrences: HashMap<String, u32>, initial_tokens: HashSet<String>) -> Self {
        Self {
            word_ocurrences,
            initial_tokens,
            tokens: HashMap::new(),
            word_tokens: HashMap::new(),
            max_heap: BinaryHeap::new(),
            tokens_id: HashMap::new(),
            id_tokens: HashMap::new(),
            vocabulary: HashMap::new(),
        }
    }

    pub fn initialize(&mut self) {
        let mut missing_tokens:Vec<String> = Vec::new();
        for(token, count) in &self.word_ocurrences {
           // List of tokens that initially form this word. 
           let mut word_tokens: Vec<TokenInfo> = Vec::new();
           for (index, char) in token.char_indices() {
                let token_str = char.to_string();
                // If the token is in the initial tokens, add it to the tokenizer's data structures.
                match self.initial_tokens.contains(&token_str) {
                    true => {
                        // Add the token to the list of tokens for this word.
                        word_tokens.push(TokenInfo::new(token_str.clone(), None::<String>));
                        match self.tokens_id.contains_key(&token_str) {
                            false => {
                                // Assign a unique ID to the token and add it to the tokenizer's data structures.
                                let token_id = self.tokens_id.len() as u32;
                                // Add the token and its ID to the tokens_id and id_tokens maps, and add it to the vocabulary.
                                self.tokens_id.insert(token_str.clone(), token_id);
                                self.id_tokens.insert(token_id, token_str.clone());
                                // Add the token to the vocabulary with its ID and no right token.
                                self.vocabulary.insert(token_id, (token_id, None));
                            },
                            true => {
                                // Token already exists, do nothing.
                            }
                        }
                        


                        // Create a position tuple for the token occurrence.
                        let position = ((index as u32, index as u32), None::<(u32, u32)>);

                        // Add the token occurrence to the tokens map, creating a new TokenOcurrences if it doesn't exist.
                        match self.tokens.get_mut(&TokenInfo::new(token_str.clone(), None::<String>)) {
                            Some(token_ocurrences) => {
                                token_ocurrences.add_ocurrence(token.clone(), *count, position);
                            },
                            None => {
                                let mut token_ocurrences = TokenOcurrences::new();
                                token_ocurrences.add_ocurrence(token.clone(), *count, position);
                                self.tokens.insert(TokenInfo::new(token_str.clone(), None::<String>), token_ocurrences);
                            }
                        }
                    },
                    false => {
                        // Saving the missing token for later analysis or reporting.
                        missing_tokens.push(token_str);
                    }
                }
            } 
            // After processing all characters in the word, store the list of tokens for this word in the word_tokens map.
            self.word_tokens.insert(token.clone(), word_tokens);
        }
        println!("Missing tokens: {:?}", missing_tokens);
    }

    pub fn start(&mut self) -> Result<(), TokenizationError> {
        self.initialize();
        
        let mut new_tokens: HashSet<TokenInfo> = HashSet::new();
        for (word, word_tokens) in self.word_tokens.iter() {
           let count: u32;
           match self.word_ocurrences.get(word) {
               Some(&ocurrences) => {
                   count = ocurrences;
               }
               None => {
                   return Err(TokenizationError::WordNotFound(word.clone()));
               }
           }
           let index: u32 = 1;
           let mut char_index: u32 = 0;
           while index < word_tokens.len() as u32 - 1 {
               let prev: &TokenInfo = &word_tokens[(index - 1) as usize];
               let current: &TokenInfo = &word_tokens[index as usize];

               let candidate_token: TokenInfo = TokenInfo::merge(prev, current);
               let candidate_token_size = candidate_token.size();
               let left_pos = (char_index, (char_index + prev.size())-1);
               let right_pos= Some((char_index + prev.size(), (char_index + prev.size() + current.size())-1));

               let position: ((u32, u32), Option<(u32, u32)>) = (left_pos, right_pos);
               // inserting the ocurrence in the new tokens map.
               new_tokens.insert(candidate_token.clone());
               self.tokens.entry(candidate_token)
                   .or_insert_with(TokenOcurrences::new)
                   .add_ocurrence(word.clone(), count, position);
               
               char_index += candidate_token_size;
           }
        }
        for token_info in new_tokens {
            match self.tokens.get(&token_info) {
                Some(token_ocurrences) => {
                    self.max_heap.push((token_ocurrences.get_ocurrences_count(), token_info));
                },
                None => {
                    return Err(TokenizationError::MissingToken(token_info.to_string()));
                }
            }
        }

       Ok(()) 
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenizer_initialization() {
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("hello".to_string(), 1);
        word_ocurrences.insert("world".to_string(), 1);

        let mut initial_tokens = HashSet::new();
        initial_tokens.insert("h".to_string());
        initial_tokens.insert("e".to_string());
        initial_tokens.insert("l".to_string());
        initial_tokens.insert("o".to_string());
        initial_tokens.insert("w".to_string());
        initial_tokens.insert("r".to_string());
        initial_tokens.insert("d".to_string());

        let mut tokenizer = BPETokenizer::new(word_ocurrences, initial_tokens);
        tokenizer.initialize();

        assert_eq!(tokenizer.word_ocurrences.len(), 2);
        assert_eq!(tokenizer.initial_tokens.len(), 7);
        assert_eq!(tokenizer.tokens.len(), 7); // Each character is a token
        assert_eq!(tokenizer.word_tokens.len(), 2); // Two words: "hello" and "world"
        assert_eq!(tokenizer.max_heap.len(), 0); // No new tokens yet
        assert_eq!(tokenizer.tokens_id.len(), 7); // Each character has an ID
        assert_eq!(tokenizer.id_tokens.len(), 7); // Each ID maps back to a character
        assert_eq!(tokenizer.vocabulary.len(), 7); // Each character is in the vocabulary

        assert_eq!(tokenizer.word_tokens.get("hello").unwrap().len(), 5); // "h", "e", "l", "l", "o"
        assert_eq!(tokenizer.word_tokens.get("world").unwrap().len(), 5); // "w", "o", "r", "l", "d"
        
        let token_ocurrences_cases : Vec<(&str, u32)> = vec![
            ("h", 1),
            ("e", 1),
            ("l", 3), // "l" appears three times in "hello"
            ("o", 2), // "o" appears in both "hello" and "world"
            ("w", 1),
            ("r", 1),
            ("d", 1),
        ];

        for (token, expected_count) in token_ocurrences_cases {
            let temp_token_info = TokenInfo::new(token.to_string(), None::<String>);
            assert!(tokenizer.tokens.contains_key(&temp_token_info));
            assert_eq!(tokenizer.tokens.get(&temp_token_info).unwrap().get_ocurrences_count(), expected_count);
        }
 
        let token_position_cases : Vec<(&str, HashMap<String, Vec<((u32, u32), Option<(u32, u32)>)>>)> = vec![
            ("h", HashMap::from([("hello".to_string(), vec![((0, 0), None::<(u32, u32)>)])])),
            ("e", HashMap::from([("hello".to_string(), vec![((1, 1), None::<(u32, u32)>)])])),
            ("l", HashMap::from([("hello".to_string(), vec![((2, 2), None::<(u32, u32)>), ((3, 3), None::<(u32, u32)>)]), ("world".to_string(), vec![((3, 3), None::<(u32, u32)>)])])),
            ("o", HashMap::from([("hello".to_string(), vec![((4, 4), None::<(u32, u32)>)]), ("world".to_string(), vec![((1, 1), None::<(u32, u32)>)])])),
            ("w", HashMap::from([("world".to_string(), vec![((0, 0), None::<(u32, u32)>)])])),
            ("r", HashMap::from([("world".to_string(), vec![((2, 2), None::<(u32, u32)>)])])),
            ("d", HashMap::from([("world".to_string(), vec![((4, 4), None::<(u32, u32)>)])])),
        ];

        for (token, expected_positions) in token_position_cases {
            let temp_token_info = TokenInfo::new(token.to_string(), None::<String>);
            assert!(tokenizer.tokens.contains_key(&temp_token_info));
            let actual_positions = tokenizer.tokens.get(&temp_token_info).unwrap().get_ocurrences();
            assert_eq!(actual_positions, &expected_positions);
        }

    }
}