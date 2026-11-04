use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::TokenizationError;

use super::BPETokenizer;

// File format (all integers little-endian u32):
//
//   magic:        4 bytes = "BPET"
//   version:      u32 = 1
//   merge_count:  u32
//   merges:       merge_count entries, in merged-ID order (256, 257, ...):
//                  left_id:  u32
//                  right_id: u32
//
// The base vocabulary (IDs 0-255, one per byte value) and the byte content of
// every learned token are reconstructed deterministically from the merge list:
// token bytes = left token bytes + right token bytes. Trailing data is rejected.

const MAGIC: &[u8; 4] = b"BPET";
const FORMAT_VERSION: u32 = 1;
// Header = 4-byte magic + 4-byte version + 4-byte merge count.
const HEADER_SIZE: usize = MAGIC.len() + 2 * std::mem::size_of::<u32>();
// Each merge stores a 4-byte left ID and a 4-byte right ID.
const MERGE_ENTRY_SIZE: usize = 2 * std::mem::size_of::<u32>();
// Learned IDs start at 256; N merges use IDs 256 through 255 + N.
const MAX_MERGE_COUNT: u32 = u32::MAX - 255;

impl BPETokenizer {
    /**
     * Serializes the learned merges into the compact model format.
     * Only the merge list is stored; every other structure (token bytes, id maps)
     * is deterministically rebuilt from it on load.
     */
    pub fn to_bytes(&self) -> Result<Vec<u8>, TokenizationError> {
        if !self.initialized {
            return Err(TokenizationError::UnexpectedError(
                "Cannot serialize a tokenizer that was not initialized.".to_string(),
            ));
        }

        // The learned merge list, indexed by merged token ID - 256.
        let merge_count = self.pair_merges.len();
        // The serialized header stores a u32 count; the same bound also ensures
        // the highest learned ID (255 + merge_count) fits in u32.
        let merge_count_u32 = u32::try_from(merge_count).map_err(|_| {
            TokenizationError::UnexpectedError("Merge list exceeds the token ID range.".to_string())
        })?;
        if merge_count_u32 > MAX_MERGE_COUNT {
            return Err(TokenizationError::UnexpectedError(
                "Merge list exceeds the token ID range.".to_string(),
            ));
        }
        let mut merge_list: Vec<(u32, u32)> = vec![(0, 0); merge_count];
        for ((left_id, right_id), merged_id) in &self.pair_merges {
            let index = match merged_id.checked_sub(256) {
                Some(index) if (index as usize) < merge_count => index as usize,
                _ => {
                    return Err(TokenizationError::UnexpectedError(format!(
                        "Merge result ID {} is out of range.",
                        merged_id
                    )));
                }
            };
            if *left_id >= *merged_id || *right_id >= *merged_id {
                return Err(TokenizationError::UnexpectedError(format!(
                    "Merge {} references part ID >= result ID.",
                    merged_id
                )));
            }
            merge_list[index] = (*left_id, *right_id);
        }

        // Check the capacity arithmetic before allocating the serialized byte buffer.
        let capacity = merge_count
            .checked_mul(MERGE_ENTRY_SIZE)
            .and_then(|entries_size| HEADER_SIZE.checked_add(entries_size))
            .ok_or_else(|| {
                TokenizationError::UnexpectedError("Serialized model is too large.".to_string())
            })?;
        let mut buffer: Vec<u8> = Vec::with_capacity(capacity);
        buffer.extend_from_slice(MAGIC);
        buffer.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buffer.extend_from_slice(&merge_count_u32.to_le_bytes());
        for (left_id, right_id) in merge_list {
            buffer.extend_from_slice(&left_id.to_le_bytes());
            buffer.extend_from_slice(&right_id.to_le_bytes());
        }
        Ok(buffer)
    }

    /**
     * Reconstructs a tokenizer from a serialized model.
     * The result is ready for encode/decode; it has no training state, and is
     * already marked as initialized so it cannot be initialized again.
     */
    pub fn from_bytes(data: &[u8]) -> Result<Self, TokenizationError> {
        let mut reader = Reader { data, pos: 0 };

        let magic = reader.read_bytes(MAGIC.len())?;
        if magic != MAGIC {
            return Err(TokenizationError::InvalidModel(
                "Bad magic, not a BPET model.".to_string(),
            ));
        }
        let version = reader.read_u32()?;
        if version != FORMAT_VERSION {
            return Err(TokenizationError::InvalidModel(format!(
                "Unsupported format version {}.",
                version
            )));
        }
        let merge_count_u32 = reader.read_u32()?;
        // The highest learned ID is 255 + merge_count; keep it in the u32 ID range.
        if merge_count_u32 > MAX_MERGE_COUNT {
            return Err(TokenizationError::InvalidModel(
                "Merge list exceeds the token ID range.".to_string(),
            ));
        }
        let merge_count = usize::try_from(merge_count_u32).map_err(|_| {
            TokenizationError::InvalidModel("Merge list is too large.".to_string())
        })?;
        // reader.pos tracks bytes consumed; after the header, the rest must be
        // exactly merge_count entries, each containing two u32 IDs.
        let remaining_bytes = reader.data.len() - reader.pos;
        let expected_merge_bytes = merge_count.checked_mul(MERGE_ENTRY_SIZE).ok_or_else(|| {
            TokenizationError::InvalidModel("Merge list is too large.".to_string())
        })?;
        if remaining_bytes != expected_merge_bytes {
            return Err(TokenizationError::InvalidModel(if remaining_bytes < expected_merge_bytes {
                "Model data is truncated.".to_string()
            } else {
                "Trailing data after the merge list.".to_string()
            }));
        }
        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(merge_count);
        for index in 0..merge_count {
            let merged_id = 256 + index as u32;
            let left_id = reader.read_u32()?;
            let right_id = reader.read_u32()?;
            if left_id >= merged_id || right_id >= merged_id {
                return Err(TokenizationError::InvalidModel(format!(
                    "Merge {} references part ID >= result ID.",
                    merged_id
                )));
            }
            merges.push((left_id, right_id));
        }
        // Rebuild the base vocabulary and the byte content of every learned token.
        let mut id_tokens: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut token_sizes: Vec<u32> = Vec::with_capacity(256 + merge_count);
        for byte in u8::MIN..=u8::MAX {
            id_tokens.insert(u32::from(byte), vec![byte]);
            token_sizes.push(1);
        }
        let mut pair_merges: HashMap<(u32, u32), u32> = HashMap::with_capacity(merge_count);
        for (index, (left_id, right_id)) in merges.into_iter().enumerate() {
            let merged_id = 256 + index as u32;
            let mut token_bytes = id_tokens
                .get(&left_id)
                .cloned()
                .ok_or_else(|| TokenizationError::InvalidModel(format!(
                    "Merge {} references unknown left ID {}.",
                    merged_id, left_id
                )))?;
            token_bytes.extend_from_slice(id_tokens.get(&right_id).ok_or_else(|| {
                TokenizationError::InvalidModel(format!(
                    "Merge {} references unknown right ID {}.",
                    merged_id, right_id
                ))
            })?);
            token_sizes.push(token_bytes.len() as u32);
            id_tokens.insert(merged_id, token_bytes);
            pair_merges.insert((left_id, right_id), merged_id);
        }
        let mut tokens_id: HashMap<Vec<u8>, u32> = HashMap::with_capacity(id_tokens.len());
        for (id, token_bytes) in &id_tokens {
            tokens_id.insert(token_bytes.clone(), *id);
        }

        Ok(Self {
            words: Vec::new(),
            word_freqs: Vec::new(),
            word_tokens: Vec::new(),
            tokens: HashMap::new(),
            max_heap: Default::default(),
            tokens_id,
            id_tokens,
            token_sizes,
            pair_merges,
            initialized: true,
        })
    }

    /**
     * Writes the serialized model to a file, replacing it if it already exists.
     */
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), TokenizationError> {
        let data = self.to_bytes()?;
        fs::write(path, data)
            .map_err(|e| TokenizationError::Io(format!("Failed to write model: {}", e)))
    }

    /**
     * Loads a tokenizer from a model file.
     */
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TokenizationError> {
        let data = fs::read(path)
            .map_err(|e| TokenizationError::Io(format!("Failed to read model: {}", e)))?;
        Self::from_bytes(&data)
    }
}

/** Cursor over the serialized bytes, used while parsing a model. */
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], TokenizationError> {
        let end = self.pos.checked_add(len).ok_or_else(|| {
            TokenizationError::InvalidModel("Model data is truncated.".to_string())
        })?;
        if end > self.data.len() {
            return Err(TokenizationError::InvalidModel(
                "Model data is truncated.".to_string(),
            ));
        }
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn read_u32(&mut self) -> Result<u32, TokenizationError> {
        let bytes = self.read_bytes(4)?;
        let mut buffer = [0u8; 4];
        buffer.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BPETokenizer, PreTokenization};

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
    fn round_trip_preserves_encoding_behavior() {
        let corpus = "the quick brown fox jumps over the lazy dog the fox";
        let original = trained_tokenizer(corpus, 15);
        let restored = BPETokenizer::from_bytes(&original.to_bytes().unwrap()).unwrap();

        let text = "the quick brown fox and an unseen word: waffles";
        assert_eq!(original.encode(text), restored.encode(text));
        assert_eq!(
            original.decode(&original.encode(text)).unwrap(),
            restored.decode(&restored.encode(text)).unwrap(),
        );
        assert_eq!(original.pair_merges, restored.pair_merges);
        assert_eq!(original.id_tokens, restored.id_tokens);
        assert_eq!(original.tokens_id, restored.tokens_id);
    }

    #[test]
    fn round_trip_preserves_non_utf8_token_bytes() {
        // "é" is two UTF-8 bytes, so merges can carry non-UTF-8 boundaries.
        let original = trained_tokenizer("héllo héllo wörld wörld", 4);
        let restored = BPETokenizer::from_bytes(&original.to_bytes().unwrap()).unwrap();

        let text = "héllo wörld";
        let ids = original.encode(text);
        assert_eq!(ids, restored.encode(text));
        assert_eq!(restored.decode(&ids).unwrap(), text.as_bytes().to_vec());
    }

    #[test]
    fn empty_model_round_trips() {
        let original = trained_tokenizer("hi hi", 0);
        let restored = BPETokenizer::from_bytes(&original.to_bytes().unwrap()).unwrap();

        assert!(restored.pair_merges.is_empty());
        assert_eq!(restored.id_tokens.len(), 256);
        assert_eq!(original.encode("hi"), restored.encode("hi"));
    }

    #[test]
    fn loaded_tokenizer_is_marked_initialized() {
        let original = trained_tokenizer("hi hi", 1);
        let mut restored = BPETokenizer::from_bytes(&original.to_bytes().unwrap()).unwrap();

        assert!(restored.initialized);
        assert!(restored.initialize().is_err());
    }

    #[test]
    fn serializing_an_uninitialized_tokenizer_is_rejected() {
        let tokenizer = BPETokenizer::new(HashMap::new());
        assert!(tokenizer.to_bytes().is_err());
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let model = trained_tokenizer("hi hi", 1).to_bytes().unwrap();
        let mut corrupt = model.clone();
        corrupt[0] = b'X';
        assert!(matches!(
            BPETokenizer::from_bytes(&corrupt),
            Err(TokenizationError::InvalidModel(_))
        ));
    }

    #[test]
    fn from_bytes_rejects_unsupported_version() {
        let model = trained_tokenizer("hi hi", 1).to_bytes().unwrap();
        let mut corrupt = model.clone();
        corrupt[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            BPETokenizer::from_bytes(&corrupt),
            Err(TokenizationError::InvalidModel(_))
        ));
    }

    #[test]
    fn from_bytes_rejects_truncated_data() {
        let model = trained_tokenizer("hello hello world", 2).to_bytes().unwrap();
        assert!(matches!(
            BPETokenizer::from_bytes(&model[..model.len() - 1]),
            Err(TokenizationError::InvalidModel(_))
        ));
        assert!(matches!(
            BPETokenizer::from_bytes(&model[..4]),
            Err(TokenizationError::InvalidModel(_))
        ));
        assert!(matches!(
            BPETokenizer::from_bytes(&[]),
            Err(TokenizationError::InvalidModel(_))
        ));
    }

    #[test]
    fn from_bytes_rejects_impossible_merge_count_before_allocation() {
        let mut model = b"BPET".to_vec();
        model.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        model.extend_from_slice(&u32::MAX.to_le_bytes());

        assert!(matches!(
            BPETokenizer::from_bytes(&model),
            Err(TokenizationError::InvalidModel(_))
        ));
    }

    #[test]
    fn from_bytes_rejects_trailing_data() {
        let model = trained_tokenizer("hi hi", 1).to_bytes().unwrap();
        let mut padded = model.clone();
        padded.push(0);
        assert!(matches!(
            BPETokenizer::from_bytes(&padded),
            Err(TokenizationError::InvalidModel(_))
        ));
    }

    #[test]
    fn from_bytes_rejects_merge_referencing_a_future_id() {
        // One merge: claim the right part is ID 300, which does not exist yet.
        let mut model = b"BPET".to_vec();
        model.extend_from_slice(&1u32.to_le_bytes()); // version
        model.extend_from_slice(&1u32.to_le_bytes()); // merge count
        model.extend_from_slice(&(104u32).to_le_bytes()); // left = 'h'
        model.extend_from_slice(&300u32.to_le_bytes()); // right = invalid
        assert!(matches!(
            BPETokenizer::from_bytes(&model),
            Err(TokenizationError::InvalidModel(_))
        ));
    }

    #[test]
    fn save_and_load_round_trip_through_a_file() {
        let original = trained_tokenizer("hello hello world", 6);
        let path = std::env::temp_dir().join(format!(
            "bpe-model-test-{}.bin",
            std::process::id()
        ));

        original.save(&path).unwrap();
        let restored = BPETokenizer::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let text = "hello world";
        assert_eq!(original.encode(text), restored.encode(text));
        assert_eq!(restored.decode_to_string(&restored.encode(text)).unwrap(), text);
    }

    #[test]
    fn load_reports_missing_file_as_io_error() {
        let path = std::env::temp_dir().join("bpe-model-does-not-exist.bin");
        assert!(matches!(
            BPETokenizer::load(&path),
            Err(TokenizationError::Io(_))
        ));
    }
}
