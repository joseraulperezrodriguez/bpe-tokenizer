use std::collections::HashMap;

use crate::TokenizationError;

/**
 * It defines the ocurrences info of a candidate token pair.
 * An occurrence is identified by the byte offset (of the pair's left part)
 * within the word; the lengths of both parts are derivable from the token
 * registry, so they are not stored per occurrence.
 */
#[derive(Clone, Debug, Default)]
pub struct TokenOcurrences {
    // Key: word id, Value: ascending byte offsets where the pair appears in the word.
    tokens_word_ocurrences: HashMap<u32, Vec<u32>>,
    // The number of ocurrences of the pair, taking into account the frequency of each word in the corpus
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

    pub fn get_ocurrences(&self) -> &HashMap<u32, Vec<u32>> {
        &self.tokens_word_ocurrences
    }

    /**
     * It adds an ocurrence of the pair at @byte_offset inside the word with id @word_id,
     * using its corpus frequency @word_frequency for the global count.
     */
    pub fn add_ocurrence(&mut self, word_id: u32, word_frequency: u64, byte_offset: u32) {
        self.tokens_word_ocurrences
            .entry(word_id)
            .or_default()
            .push(byte_offset);
        self.count_ocurrences += word_frequency;
    }

    /**
     * It replaces the previous offsets of the given word with a new list of offsets.
     */
    pub fn replace_positions(
        &mut self,
        word_id: u32,
        word_ocurrencies: u64,
        new_offsets: Vec<u32>,
    ) -> Result<(), TokenizationError> {
        let prev_count = match self.tokens_word_ocurrences.get(&word_id) {
            Some(prev_offsets) => prev_offsets.len() as u64,
            None => {
                return Err(TokenizationError::UnexpectedError(format!(
                    "Word id {} not found in the ocurrences list.",
                    word_id
                )));
            }
        };
        self.count_ocurrences -= prev_count * word_ocurrencies;
        self.count_ocurrences += new_offsets.len() as u64 * word_ocurrencies;

        self.tokens_word_ocurrences.remove(&word_id);
        if !new_offsets.is_empty() {
            self.tokens_word_ocurrences.insert(word_id, new_offsets);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_token_ocurrences() {
        let mut token_ocurrences = TokenOcurrences::new();
        // Word id 7, offsets 0 and 1.
        token_ocurrences.add_ocurrence(7, 1, 0);
        token_ocurrences.add_ocurrence(7, 1, 1);
        assert_eq!(token_ocurrences.get_ocurrences_count(), 2);

        let word_ocurrences = token_ocurrences.get_ocurrences().get(&7);
        assert!(word_ocurrences.is_some());
        assert_eq!(word_ocurrences.unwrap().as_slice(), &[0, 1]);
    }

    #[test]
    fn add_ocurrence_scales_counts_by_word_frequency() {
        let mut token_ocurrences = TokenOcurrences::new();
        token_ocurrences.add_ocurrence(3, 5, 0);
        token_ocurrences.add_ocurrence(3, 5, 4);
        assert_eq!(token_ocurrences.get_ocurrences_count(), 10);
    }

    #[test]
    fn test_replace_positions() {
        let mut ocurrences = TokenOcurrences::new();
        ocurrences.add_ocurrence(2, 3, 0);
        ocurrences.add_ocurrence(2, 3, 1);
        ocurrences.add_ocurrence(2, 3, 2);

        let res = ocurrences.replace_positions(2, 3, vec![0]);
        assert!(res.is_ok());
        assert_eq!(ocurrences.get_ocurrences_count(), 3);
        assert_eq!(ocurrences.get_ocurrences().get(&2).unwrap().as_slice(), &[0]);
    }

    #[test]
    fn replace_positions_drops_empty_word_entries() {
        let mut ocurrences = TokenOcurrences::new();
        ocurrences.add_ocurrence(2, 3, 0);
        ocurrences.add_ocurrence(2, 3, 1);

        ocurrences.replace_positions(2, 3, vec![]).unwrap();

        assert_eq!(ocurrences.get_ocurrences_count(), 0);
        assert!(!ocurrences.get_ocurrences().contains_key(&2));
    }

    #[test]
    fn replace_positions_rejects_an_unknown_word() {
        let mut ocurrences = TokenOcurrences::new();
        ocurrences.add_ocurrence(2, 3, 0);

        assert!(ocurrences.replace_positions(9, 3, vec![0]).is_err());
    }
}
