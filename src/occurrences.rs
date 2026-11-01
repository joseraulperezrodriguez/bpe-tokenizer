use std::collections::HashMap;

use crate::{TokenizationError};

/**
 * It defines where an ocurrence is located inside a word.
 */
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OcurrencePosition {
  // Byte offset within the UTF-8 word.
  byte_offset: usize,
  // Number of bytes in the token.
  token_length: usize,
}

impl OcurrencePosition {
    pub fn new(byte_offset: usize, token_length: usize) -> Self {
      Self { byte_offset, token_length }
    }

    pub fn get_byte_offset(&self) -> &usize {
      &self.byte_offset
    }

    pub fn get_token_length(&self) -> &usize {
      &self.token_length
    }
}

/**
 * It defines the ocurrences info of a token.
 */
#[derive(Clone, Debug, Default)]
pub struct TokenOcurrences {
    // Key: word, Value: positions within that word where the token appeared.
    // The tokens are stored as a pair; the parts that forms the token.
    tokens_word_ocurrences: HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>,
    // The number of ocurrences of the associated token, taking into account the frequency of each word in the corpus
    // formule for word {w} with frequency freq(w) = len(tokens_word_ocurrences[w]) * freq(w).
    count_ocurrences: u64,
}

impl TokenOcurrences {
    pub fn new() -> Self {
        Self {
            tokens_word_ocurrences: HashMap::new(),
            count_ocurrences: 0,
        }
    }

    pub fn get_ocurrences_count(&self) -> u64 {
        self.count_ocurrences
    }

    pub fn get_ocurrences(
        &self,
    ) -> &HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>> {
        &self.tokens_word_ocurrences
    }

    /**
     * It adds an ocurrence of the tokenn described by @position, to the list of ocurrences for the given @word
     * Using the @word_frequency count.
     */
    pub fn add_ocurrence(
        &mut self,
        word: String,
        word_frequency: u64,
        position: (OcurrencePosition, Option<OcurrencePosition>),
    ) {
        self.tokens_word_ocurrences
            .entry(word)
            .or_default()
            .push(position);
        self.count_ocurrences += word_frequency;
    }

    /**
     * We completely the previous existing ocurrences of the given word, with a new list of ocurrences. 
     * */
    pub fn replace_positions(&mut self, word: String, word_ocurrencies: u64, new_pos: Vec<(OcurrencePosition, Option<OcurrencePosition>)>) -> Result<(), TokenizationError> {

       let prev_count = match self.tokens_word_ocurrences.get(&word) {
         Some(prev_positions) => {
            prev_positions.len() as u64
         }
         None => {
            return Err(TokenizationError::WordNotFound(word));
         }
       };
       self.count_ocurrences -= prev_count * word_ocurrencies;
       self.count_ocurrences += new_pos.len() as u64 * word_ocurrencies;

       self.tokens_word_ocurrences.remove(&word);
       if new_pos.len() > 0 {
         self.tokens_word_ocurrences.insert(word,  new_pos);
       }
       return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_token_ocurrences() {
        let mut token_ocurrences = TokenOcurrences::new();
        token_ocurrences.add_ocurrence(
            " word".to_string(),
            1,
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
        );
        token_ocurrences.add_ocurrence(
            " word".to_string(),
            1,
            (
                OcurrencePosition::new(1, 1),
                Some(OcurrencePosition::new(2, 1)),
            ),
        );
        assert_eq!(token_ocurrences.get_ocurrences_count(), 2);

        let word_ocurrences = token_ocurrences.get_ocurrences().get(" word");
        assert!(word_ocurrences.is_some());
        assert_eq!(
            word_ocurrences.unwrap().as_slice(),
            &[
                (
                    OcurrencePosition::new(0, 1),
                    Some(OcurrencePosition::new(1, 1)),
                ),
                (
                    OcurrencePosition::new(1, 1),
                    Some(OcurrencePosition::new(2, 1)),
                ),
            ]
        );
    }

    #[test]
    fn test_replace_positions() {
        let mut ocurrences = TokenOcurrences::new();
        let word = "wow".to_string();
        ocurrences.add_ocurrence(word.clone(), 3, (OcurrencePosition::new(0, 1), None));
        ocurrences.add_ocurrence(word.clone(), 3, (OcurrencePosition::new(1, 1), None));
        ocurrences.add_ocurrence(word.clone(), 3, (OcurrencePosition::new(2, 1), None));

        let new_pos = vec![(OcurrencePosition::new(0, 1), None::<OcurrencePosition>)];

        let res = ocurrences.replace_positions(word.clone(), 3, new_pos); 
        assert!(res.is_ok());
        assert!(ocurrences.get_ocurrences_count() == 3);
    }
}