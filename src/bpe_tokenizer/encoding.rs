use std::collections::HashMap;

use crate::{PreTokenization, TokenizationError};

use super::BPETokenizer;

impl BPETokenizer {
    /**
     * Encodes a piece of text into token ids.
     * The text is pre-tokenized into words, and each word is encoded by repeatedly
     * applying the learned merges in the order they were learned.
     */
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // A per-call cache keeps the existing semantics: callers of encode()
        // get no reuse across calls.
        let mut word_cache: HashMap<String, Vec<u32>> = HashMap::new();
        self.encode_with_cache(text, &mut word_cache)
    }

    /**
     * Encodes a piece of text into token ids, reusing a persistent per-word cache.
     *
     * The cache maps each pre-token to its encoded ids, so a word repeated across
     * calls (the norm in a bulk corpus export) is encoded once. Sharing one cache
     * over a whole corpus is safe: encoding is a pure function of the merge table,
     * which is immutable after loading.
     *
     * Keys are owned because the pre-token slices only live as long as @text.
     */
    pub fn encode_with_cache(
        &self,
        text: &str,
        cache: &mut HashMap<String, Vec<u32>>,
    ) -> Vec<u32> {
        let words = PreTokenization::split(text);
        let mut ids: Vec<u32> = Vec::new();
        for word in words {
            if !cache.contains_key(word) {
                let encoded = self.encode_word(word.as_bytes());
                cache.insert(word.to_string(), encoded);
            }
            ids.extend_from_slice(&cache[word]);
        }
        ids
    }

    /**
     * Encodes a single word by starting from its raw bytes and repeatedly merging
     * the adjacent pair with the lowest learned merge id until no learned pair remains.
     */
    fn encode_word(&self, word: &[u8]) -> Vec<u32> {
        let mut ids: Vec<u32> = word.iter().map(|byte| u32::from(*byte)).collect();
        while ids.len() > 1 {
            // Find the adjacent pair with the lowest merge id, which was learned first.
            let mut best_pair: Option<((u32, u32), u32)> = None;
            for index in 0..ids.len() - 1 {
                let pair = (ids[index], ids[index + 1]);
                if let Some(&merged_id) = self.pair_merges.get(&pair) {
                    if best_pair.map_or(true, |(_, best_id)| merged_id < best_id) {
                        best_pair = Some((pair, merged_id));
                    }
                }
            }
            let Some((pair, _)) = best_pair else {
                break;
            };
            // Merge every occurrence of the winning pair, left to right.
            // After merging at index i we skip the merged element so overlapping
            // occurrences (e.g. "aaa" for pair "aa") are not merged twice.
            let mut merged_ids: Vec<u32> = Vec::with_capacity(ids.len());
            let mut index = 0;
            while index < ids.len() {
                if index + 1 < ids.len() && (ids[index], ids[index + 1]) == pair {
                    merged_ids.push(self.pair_merges[&pair]);
                    index += 2;
                } else {
                    merged_ids.push(ids[index]);
                    index += 1;
                }
            }
            ids = merged_ids;
        }
        ids
    }

    /**
     * Decodes a list of token ids back into the raw bytes they represent.
     */
    pub fn decode(&self, ids: &[u32]) -> Result<Vec<u8>, TokenizationError> {
        let mut bytes: Vec<u8> = Vec::new();
        for id in ids {
            match self.id_tokens.get(id) {
                Some(token_bytes) => bytes.extend_from_slice(token_bytes),
                // encode() can emit base-byte IDs before initialize() populates id_tokens.
                None if *id <= u8::MAX as u32 => bytes.push(*id as u8),
                None => return Err(TokenizationError::UnknownTokenId(*id)),
            }
        }
        Ok(bytes)
    }

    /**
     * Decodes a list of token ids into a string.
     * Invalid UTF-8 sequences are replaced with the Unicode replacement character,
     * so decoding is lossy for non-UTF-8 token boundaries.
     */
    pub fn decode_to_string(&self, ids: &[u32]) -> Result<String, TokenizationError> {
        let bytes = self.decode(ids)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BPETokenizer, PreTokenization};

    // Trains a tokenizer on the given corpus for @merge_count merges.
    fn trained_tokenizer(corpus: &str, merge_count: usize) -> BPETokenizer {
        let mut word_ocurrences = HashMap::new();
        PreTokenization::count_tokens(corpus, &mut word_ocurrences);
        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();
        for _ in 0..merge_count {
            tokenizer.next_token().unwrap();
        }
        tokenizer
    }

    #[test]
    fn encode_with_cache_agrees_with_encode() {
        let tokenizer = trained_tokenizer("the quick brown fox", 10);

        let text = "the quick brown cat jumps over the lazy dog";
        let mut cache = HashMap::new();
        let cached_ids = tokenizer.encode_with_cache(text, &mut cache);
        assert_eq!(cached_ids, tokenizer.encode(text));
    }

    #[test]
    fn encode_with_cache_hits_the_cache_for_repeated_words() {
        let tokenizer = trained_tokenizer("hello world", 5);

        let mut cache = HashMap::new();
        tokenizer.encode_with_cache("hello world", &mut cache);
        let cache_size = cache.len();
        assert!(cache_size > 0);

        // The second call must reuse the entries: no new cache words, and the
        // same ids come back out.
        let ids = tokenizer.encode_with_cache("hello world", &mut cache);
        assert_eq!(cache.len(), cache_size);
        assert_eq!(tokenizer.decode_to_string(&ids).unwrap(), "hello world");

        // A new word extends the cache by exactly its pre-tokens.
        let words_before = cache.len();
        tokenizer.encode_with_cache("hello there world", &mut cache);
        assert!(cache.len() > words_before);
    }

    #[test]
    fn encode_untrained_tokenizer_falls_back_to_raw_bytes() {
        let mut word_ocurrences = HashMap::new();
        PreTokenization::count_tokens("hi", &mut word_ocurrences);
        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();

        let ids = tokenizer.encode("hi");
        assert_eq!(ids, vec![u32::from(b'h'), u32::from(b'i')]);
        assert_eq!(tokenizer.decode_to_string(&ids).unwrap(), "hi");
    }

    #[test]
    fn decode_accepts_base_byte_ids_before_initialization() {
        let tokenizer = BPETokenizer::new(HashMap::new());

        assert_eq!(tokenizer.decode(&[104, 105]).unwrap(), b"hi");
        assert!(matches!(
            tokenizer.decode(&[256]),
            Err(crate::TokenizationError::UnknownTokenId(256))
        ));
    }

    #[test]
    fn encode_applies_learned_merges() {
        // "lo" appears 3 times, "ow" 3 times (tie broken lexicographically, so
        // "lo" is learned first), and then "low" is learned as the second merge.
        let tokenizer = trained_tokenizer("low low lower", 2);

        let ids = tokenizer.encode("low");
        // The whole word is a learned token after the two merges.
        let merged_bytes: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| tokenizer.decode(&[*id]).unwrap())
            .collect();
        assert_eq!(merged_bytes, vec![b"low".to_vec()]);

        // "lower" starts with the same merges, so its prefix encodes as "low".
        let ids = tokenizer.encode("lower");
        let merged_bytes: Vec<Vec<u8>> = ids
            .iter()
            .map(|id| tokenizer.decode(&[*id]).unwrap())
            .collect();
        assert_eq!(
            merged_bytes,
            vec![b"low".to_vec(), b"e".to_vec(), b"r".to_vec()]
        );
    }

    #[test]
    fn encode_handles_words_not_seen_during_training() {
        let tokenizer = trained_tokenizer("low low lower", 2);

        // "newest" was never in the corpus, but any learned merges that apply are used.
        let ids = tokenizer.encode("lower");
        let decoded = tokenizer.decode_to_string(&ids).unwrap();
        assert_eq!(decoded, "lower");
    }

    #[test]
    fn encode_round_trips_text_with_unseen_words() {
        let corpus = "the quick brown fox jumps over the lazy dog";
        let tokenizer = trained_tokenizer(corpus, 20);

        let text = "the quick brown cat";
        let ids = tokenizer.encode(text);
        assert_eq!(tokenizer.decode_to_string(&ids).unwrap(), text);
    }

    #[test]
    fn encode_merges_overlapping_pairs_left_to_right() {
        // With merges for "aa" learned from "aaa", overlapping occurrences are
        // merged left to right: "aaa" becomes "aa"+"a", never a triple merge.
        let tokenizer = trained_tokenizer("aaaa aaaa", 1);

        let ids = tokenizer.encode("aaaa");
        let parts: Vec<String> = ids
            .iter()
            .map(|id| tokenizer.decode_to_string(&[*id]).unwrap())
            .collect();
        assert_eq!(parts, vec!["aa".to_string(), "aa".to_string()]);
    }

    #[test]
    fn encode_repeated_words_are_consistent() {
        let tokenizer = trained_tokenizer("hello hello world", 5);

        let ids = tokenizer.encode("hello world hello");
        let roundtrip = tokenizer.decode_to_string(&ids).unwrap();
        assert_eq!(roundtrip, "hello world hello");
    }

    #[test]
    fn decode_returns_raw_bytes() {
        let tokenizer = trained_tokenizer("hi hi", 1);
        let ids = tokenizer.encode("hi");
        assert_eq!(tokenizer.decode(&ids).unwrap(), b"hi".to_vec());
    }

    #[test]
    fn decode_rejects_unknown_token_ids() {
        let mut word_ocurrences = HashMap::new();
        PreTokenization::count_tokens("hi", &mut word_ocurrences);
        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();

        assert!(tokenizer.decode(&[0]).is_ok());
        let err = tokenizer.decode(&[999_999]).unwrap_err();
        assert!(matches!(
            err,
            crate::TokenizationError::UnknownTokenId(999_999)
        ));
    }
}
