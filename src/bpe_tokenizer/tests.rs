#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TokenInfo, TokenOcurrences, TokenizationError};
    use std::cmp::Reverse;
    use std::collections::HashSet;

    fn byte_id(byte: u8) -> u32 {
        u32::from(byte)
    }

    fn pair_of_bytes(left: u8, right: u8) -> (u32, u32) {
        (byte_id(left), byte_id(right))
    }

    /**
     * A tokenizer with no corpus words but with the 256 base byte tokens
     * registered, so pairs of byte ids resolve their sizes.
     */
    fn byte_ready_tokenizer() -> BPETokenizer {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        tokenizer.setup();
        tokenizer
    }

    /**
     * Registers the merged token for the given pair, mirroring what
     * next_token does before merging: assigns the next id, stores the bytes,
     * size, and the pair_merges entry.
     */
    fn register_merged_token(tokenizer: &mut BPETokenizer, pair: (u32, u32)) -> u32 {
        let token_id = tokenizer.id_tokens.len() as u32;
        let mut token_bytes = tokenizer.id_tokens[&pair.0].clone();
        token_bytes.extend_from_slice(&tokenizer.id_tokens[&pair.1]);
        tokenizer.tokens_id.insert(token_bytes.clone(), token_id);
        tokenizer.id_tokens.insert(token_id, token_bytes.clone());
        tokenizer.token_sizes.push(token_bytes.len() as u32);
        tokenizer.pair_merges.insert(pair, token_id);
        token_id
    }

    fn pair_info(left: u8, right: u8) -> TokenInfo {
        TokenInfo::new(vec![left], Some(vec![right]))
    }

    #[test]
    fn test_bpe_tokenizer_initial_setup() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([
            ("hello".to_string(), 1),
            ("world".to_string(), 1),
        ]));
        tokenizer.setup();

        // The word registry: sorted words with their frequencies.
        assert_eq!(
            tokenizer.words.iter().map(|w| &**w).collect::<Vec<_>>(),
            vec!["hello", "world"]
        );
        assert_eq!(tokenizer.word_freqs, vec![1, 1]);

        // The token id registries: the 256 base byte tokens.
        assert_eq!(tokenizer.tokens_id.len(), 256);
        assert_eq!(tokenizer.id_tokens.len(), 256);
        assert_eq!(tokenizer.token_sizes.len(), 256);
        for byte in u8::MIN..=u8::MAX {
            assert_eq!(tokenizer.tokens_id.get(&vec![byte]), Some(&u32::from(byte)));
            assert_eq!(tokenizer.token_size(u32::from(byte)), 1);
        }

        // Each word starts as one token id per byte.
        assert_eq!(
            tokenizer.word_tokens,
            vec![
                vec![104, 101, 108, 108, 111], // "hello"
                vec![119, 111, 114, 108, 100], // "world"
            ]
        );

        // No candidate pairs exist until quickoff runs.
        assert!(tokenizer.tokens.is_empty());
        assert!(tokenizer.max_heap.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_quickoff() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([
            ("hello".to_string(), 1),
            (" worlod".to_string(), 1),
        ]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        // hello: (h,e) (e,l) (l,l) (l,o); " worlod": (" ","w") (w,o) (o,r) (r,l) (l,o) (o,d)
        // with (l,o) shared: 9 distinct pairs.
        assert_eq!(tokenizer.tokens.len(), 9);

        // Check that the candidates have been added to the max_heap with
        // matching counts.
        assert_eq!(tokenizer.max_heap.len(), 9);
        for (count, Reverse(token_info)) in &tokenizer.max_heap {
            let pair = tokenizer.pair_of(token_info).unwrap();
            assert_eq!(
                tokenizer.tokens[&pair].get_ocurrences_count(),
                *count,
                "heap count mismatch for {:?}",
                token_info.to_bytes()
            );
        }

        let (count, Reverse(token_info)) = tokenizer.max_heap.pop().unwrap();
        assert_eq!(count, 2);
        assert_eq!(token_info.to_bytes(), b"lo");

        let (count, Reverse(token_info)) = tokenizer.max_heap.pop().unwrap();
        assert_eq!(count, 1);
        assert_eq!(token_info.to_bytes(), b" w");
    }

    #[test]
    fn setup_splits_unicode_words_into_utf8_bytes() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("é".to_string(), 1)]));
        tokenizer.setup();

        assert_eq!(tokenizer.word_tokens, vec![vec![0xC3, 0xA9]]);
        assert_eq!(tokenizer.tokens_id.get(&vec![0xC3]), Some(&195));
        assert_eq!(tokenizer.tokens_id.get(&vec![0xA9]), Some(&169));
    }

    #[test]
    fn setup_with_empty_corpus_initializes_only_the_byte_vocabulary() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        tokenizer.setup();

        assert_eq!(tokenizer.tokens_id.len(), 256);
        assert_eq!(tokenizer.id_tokens.len(), 256);
        assert!(tokenizer.tokens.is_empty());
        assert!(tokenizer.word_tokens.is_empty());
    }

    #[test]
    fn setup_ignores_zero_frequency_words() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 0)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        // The word is interned but has no token sequence.
        assert_eq!(tokenizer.words.len(), 1);
        assert_eq!(tokenizer.word_tokens, vec![Vec::<u32>::new()]);
        assert!(tokenizer.tokens.is_empty());
        assert!(tokenizer.max_heap.is_empty());
    }

    #[test]
    fn learned_token_ids_start_after_the_256_byte_tokens() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        tokenizer.initialize().unwrap();
        let (token_id, token) = tokenizer.next_token().unwrap().unwrap();

        assert_eq!(token_id, 256);
        assert_eq!(token.to_bytes(), b"ab");
        assert_eq!(tokenizer.tokens_id.get(b"ab".as_slice()), Some(&256));
    }

    #[test]
    fn insert_new_pair_tracks_occurrences_and_deduplicates_candidates() {
        let mut tokens = HashMap::new();
        let mut new_pairs = HashSet::new();
        let pair = pair_of_bytes(b'a', b'b');

        BPETokenizer::insert_new_pair(pair, &mut new_pairs, &mut tokens, 0, 3, 0);
        BPETokenizer::insert_new_pair(pair, &mut new_pairs, &mut tokens, 0, 3, 0);

        assert_eq!(new_pairs.len(), 1);
        assert!(new_pairs.contains(&pair));
        assert_eq!(tokens[&pair].get_ocurrences_count(), 6);
        assert_eq!(tokens[&pair].get_ocurrences()[&0], vec![0, 0]);
    }

    #[test]
    fn insert_new_pair_does_not_mark_an_existing_candidate_as_new() {
        let pair = pair_of_bytes(b'a', b'b');
        let mut tokens = HashMap::new();
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(0, 2, 0);
        tokens.insert(pair, occurrences);
        let mut new_pairs = HashSet::new();

        BPETokenizer::insert_new_pair(pair, &mut new_pairs, &mut tokens, 0, 3, 0);

        assert!(new_pairs.is_empty());
        assert_eq!(tokens[&pair].get_ocurrences_count(), 5);
    }

    #[test]
    fn pair_of_resolves_registered_pairs() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        tokenizer.initialize().unwrap();

        assert_eq!(tokenizer.pair_of(&pair_info(b'a', b'b')), Some((97, 98)));
    }

    #[test]
    fn pair_of_rejects_unknown_bytes_or_missing_right_part() {
        let tokenizer = byte_ready_tokenizer();

        // Unknown left or right bytes do not resolve.
        assert!(tokenizer
            .pair_of(&TokenInfo::new(b"missing".to_vec(), Some(b"a".to_vec())))
            .is_none());
        assert!(tokenizer
            .pair_of(&TokenInfo::new(b"a".to_vec(), Some(b"missing".to_vec())))
            .is_none());
        // A single token without a right part is not a pair.
        assert!(tokenizer.pair_of(&TokenInfo::new(b"a".to_vec(), None)).is_none());
    }

    #[test]
    fn quickoff_succeeds_for_empty_corpus() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        tokenizer.setup();

        assert!(tokenizer.quickoff().is_ok());
        assert!(tokenizer.max_heap.is_empty());
    }

    #[test]
    fn quickoff_errors_when_word_has_no_byte_tokens() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("".to_string(), 1)]));
        tokenizer.setup();

        assert!(tokenizer.quickoff().is_err());
    }

    #[test]
    fn merge_winning_token_and_generate_new_ones_updates_word_tokens() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        tokenizer.initialize().unwrap();
        let pair = pair_of_bytes(b'a', b'b');
        register_merged_token(&mut tokenizer, pair);

        tokenizer
            .merge_winning_token_and_generate_new_ones(pair)
            .unwrap();

        assert_eq!(tokenizer.word_tokens, vec![vec![256]]);
    }

    #[test]
    fn merge_winning_token_generates_candidates_on_both_sides() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("xaby".to_string(), 2)]));
        tokenizer.initialize().unwrap();
        let pair = pair_of_bytes(b'a', b'b');
        register_merged_token(&mut tokenizer, pair);

        tokenizer
            .merge_winning_token_and_generate_new_ones(pair)
            .unwrap();

        assert_eq!(tokenizer.word_tokens, vec![vec![120, 256, 121]]); // x, ab, y
        let left_candidate = (120, 256); // x + ab
        let right_candidate = (256, 121); // ab + y
        assert_eq!(
            tokenizer.tokens[&left_candidate].get_ocurrences_count(),
            2
        );
        assert_eq!(
            tokenizer.tokens[&right_candidate].get_ocurrences_count(),
            2
        );
        assert!(tokenizer
            .max_heap
            .iter()
            .any(|(_, Reverse(info))| tokenizer.pair_of(info) == Some(left_candidate)));
        assert!(tokenizer
            .max_heap
            .iter()
            .any(|(_, Reverse(info))| tokenizer.pair_of(info) == Some(right_candidate)));
    }

    #[test]
    fn merge_winning_token_errors_when_candidate_is_missing() {
        let mut tokenizer = byte_ready_tokenizer();
        let pair = pair_of_bytes(b'a', b'b');
        register_merged_token(&mut tokenizer, pair);

        assert!(matches!(
            tokenizer.merge_winning_token_and_generate_new_ones(pair),
            Err(TokenizationError::UnexpectedError(_))
        ));
    }

    #[test]
    fn merge_winning_token_errors_when_word_tokens_are_missing() {
        let mut tokenizer = byte_ready_tokenizer();
        let pair = pair_of_bytes(b'a', b'b');
        register_merged_token(&mut tokenizer, pair);
        // The pair has occurrences pointing at word 0, but no word was interned.
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(0, 1, 0);
        tokenizer.tokens.insert(pair, occurrences);

        assert!(matches!(
            tokenizer.merge_winning_token_and_generate_new_ones(pair),
            Err(TokenizationError::UnexpectedError(_))
        ));
    }

    #[test]
    fn get_tokens_to_update_tracks_overlap_with_winning_token() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abc".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = pair_of_bytes(b'b', b'c');
        let updates = tokenizer.get_tokens_to_update(winning).unwrap();
        let adjacent = pair_of_bytes(b'a', b'b');

        assert!(updates.contains_key(&adjacent));
        assert_eq!(updates[&adjacent][&0], vec![0]);
    }

    #[test]
    fn get_tokens_to_update_accumulates_repeated_adjacent_candidates() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("xabyxaby".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = pair_of_bytes(b'a', b'b');
        let updates = tokenizer.get_tokens_to_update(winning).unwrap();
        let left_adjacent = pair_of_bytes(b'x', b'a');
        let right_adjacent = pair_of_bytes(b'b', b'y');

        assert_eq!(updates[&left_adjacent][&0], vec![0, 4]);
        assert_eq!(updates[&right_adjacent][&0], vec![2, 6]);
    }

    #[test]
    fn get_tokens_to_update_returns_empty_when_winner_has_no_neighbors() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = pair_of_bytes(b'a', b'b');
        assert!(tokenizer.get_tokens_to_update(winning).unwrap().is_empty());
    }

    #[test]
    fn get_tokens_to_update_errors_for_missing_winner() {
        let tokenizer = byte_ready_tokenizer();
        let missing = pair_of_bytes(b'a', b'b');

        assert!(matches!(
            tokenizer.get_tokens_to_update(missing),
            Err(TokenizationError::MissingToken(_))
        ));
    }

    #[test]
    fn get_tokens_to_update_errors_when_word_tokens_are_missing() {
        let mut tokenizer = byte_ready_tokenizer();
        let winning = pair_of_bytes(b'a', b'b');
        // The winner has occurrences for word 0, but no word was interned.
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(0, 1, 0);
        tokenizer.tokens.insert(winning, occurrences);

        assert!(matches!(
            tokenizer.get_tokens_to_update(winning),
            Err(TokenizationError::UnexpectedError(_))
        ));
    }

    #[test]
    fn update_tokens_removes_overlapping_positions() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abc".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();

        let winning = pair_of_bytes(b'b', b'c');
        let updates = tokenizer.get_tokens_to_update(winning).unwrap();
        tokenizer.update_tokens(&updates).unwrap();

        let adjacent = pair_of_bytes(b'a', b'b');
        assert_eq!(tokenizer.tokens[&adjacent].get_ocurrences_count(), 0);
        assert!(!tokenizer.tokens[&adjacent].get_ocurrences().contains_key(&0));
    }

    #[test]
    fn update_tokens_preserves_positions_not_requested_for_removal() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abab".to_string(), 2)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();
        let adjacent = pair_of_bytes(b'a', b'b');
        assert_eq!(
            tokenizer.tokens[&adjacent].get_ocurrences_count(),
            4
        );
        let to_remove = HashMap::from([(adjacent, HashMap::from([(0, vec![0])]))]);

        tokenizer.update_tokens(&to_remove).unwrap();

        assert_eq!(tokenizer.tokens[&adjacent].get_ocurrences()[&0], vec![2]);
        assert_eq!(tokenizer.tokens[&adjacent].get_ocurrences_count(), 2);
    }

    #[test]
    fn update_tokens_ignores_a_candidate_absent_from_token_map() {
        let mut tokenizer = byte_ready_tokenizer();
        let absent = pair_of_bytes(b'a', b'b');
        let to_remove = HashMap::from([(absent, HashMap::from([(0, vec![0])]))]);

        assert!(tokenizer.update_tokens(&to_remove).is_ok());
    }

    #[test]
    fn update_tokens_errors_when_requested_word_has_no_occurrence_entry() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        tokenizer.setup();
        tokenizer.quickoff().unwrap();
        // Candidate ("a","b") exists for word 0; word 9 never had occurrences.
        let to_remove = HashMap::from([(
            pair_of_bytes(b'a', b'b'),
            HashMap::from([(9, vec![0])]),
        )]);

        assert!(tokenizer.update_tokens(&to_remove).is_err());
    }

    #[test]
    fn find_and_insert_adyacent_tokens_registers_left_overlap() {
        let tokenizer = byte_ready_tokenizer();
        let mut map = HashMap::new();
        let winning = pair_of_bytes(b'b', b'c');
        let word_tokens = vec![byte_id(b'a'), byte_id(b'b'), byte_id(b'c')];

        tokenizer
            .find_and_insert_adyacent_tokens(&mut map, winning, 0, &word_tokens, &[1])
            .unwrap();

        let adjacent = pair_of_bytes(b'a', b'b');
        assert_eq!(map[&adjacent][&0], vec![0]);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_registers_right_overlap() {
        let tokenizer = byte_ready_tokenizer();
        let mut map = HashMap::new();
        let winning = pair_of_bytes(b'b', b'c');
        let word_tokens = vec![
            byte_id(b'a'),
            byte_id(b'b'),
            byte_id(b'c'),
            byte_id(b'd'),
        ];

        tokenizer
            .find_and_insert_adyacent_tokens(&mut map, winning, 0, &word_tokens, &[1])
            .unwrap();

        let left_adjacent = pair_of_bytes(b'a', b'b');
        let adjacent = pair_of_bytes(b'c', b'd');
        assert_eq!(map[&left_adjacent][&0], vec![0]);
        assert_eq!(map[&adjacent][&0], vec![2]);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_uses_byte_offsets_for_multibyte_tokens() {
        let mut tokenizer = byte_ready_tokenizer();
        // Register multibyte tokens: "aa", "cc".
        let aa = register_merged_token(&mut tokenizer, pair_of_bytes(b'a', b'a'));
        let cc = register_merged_token(&mut tokenizer, pair_of_bytes(b'c', b'c'));

        let mut map = HashMap::new();
        // Word: [aa][b][cc][d] -> "aabccd"
        let winning = (byte_id(b'b'), cc);
        let word_tokens = vec![aa, byte_id(b'b'), cc, byte_id(b'd')];

        // The winner "bcc" starts at byte 2.
        tokenizer
            .find_and_insert_adyacent_tokens(&mut map, winning, 0, &word_tokens, &[2])
            .unwrap();

        let left_adjacent = (aa, byte_id(b'b')); // "aa"+"b" starts at byte 0
        let right_adjacent = (cc, byte_id(b'd')); // "cc"+"d" starts at byte 3
        assert_eq!(map[&left_adjacent][&0], vec![0]);
        assert_eq!(map[&right_adjacent][&0], vec![3]);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_handles_winner_at_word_start() {
        let tokenizer = byte_ready_tokenizer();
        let mut map = HashMap::new();
        let winning = pair_of_bytes(b'a', b'b');
        let word_tokens = vec![byte_id(b'a'), byte_id(b'b'), byte_id(b'c')];

        tokenizer
            .find_and_insert_adyacent_tokens(&mut map, winning, 0, &word_tokens, &[0])
            .unwrap();

        let adjacent = pair_of_bytes(b'b', b'c');
        assert_eq!(map[&adjacent][&0], vec![1]);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_handles_winner_at_word_end() {
        let tokenizer = byte_ready_tokenizer();
        let mut map = HashMap::new();
        let winning = pair_of_bytes(b'b', b'c');
        let word_tokens = vec![byte_id(b'a'), byte_id(b'b'), byte_id(b'c')];

        tokenizer
            .find_and_insert_adyacent_tokens(&mut map, winning, 0, &word_tokens, &[1])
            .unwrap();

        let adjacent = pair_of_bytes(b'a', b'b');
        assert_eq!(map[&adjacent][&0], vec![0]);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn find_and_insert_adyacent_tokens_skips_overlapping_winner_positions() {
        let tokenizer = byte_ready_tokenizer();
        let mut map = HashMap::new();
        let winning = pair_of_bytes(b'a', b'a');
        let word_tokens = vec![byte_id(b'a'), byte_id(b'a'), byte_id(b'a')];

        tokenizer
            .find_and_insert_adyacent_tokens(&mut map, winning, 0, &word_tokens, &[0, 1])
            .unwrap();

        assert!(map.is_empty());
    }

    #[test]
    fn find_and_insert_adyacent_tokens_accepts_no_occurrences() {
        let tokenizer = byte_ready_tokenizer();
        let mut map = HashMap::new();
        let winning = pair_of_bytes(b'a', b'b');
        let word_tokens = vec![byte_id(b'a'), byte_id(b'b')];

        tokenizer
            .find_and_insert_adyacent_tokens(&mut map, winning, 0, &word_tokens, &[])
            .unwrap();

        assert!(map.is_empty());
    }

    #[test]
    fn get_winning_token_skips_stale_heap_entries() {
        let mut tokenizer = byte_ready_tokenizer();
        let stale = pair_of_bytes(b'a', b'b');
        let valid = pair_of_bytes(b'c', b'd');

        tokenizer
            .tokens
            .insert(stale, TokenOcurrences::new());
        let mut valid_ocurrences = TokenOcurrences::new();
        valid_ocurrences.add_ocurrence(0, 5, 0);
        tokenizer.tokens.insert(valid, valid_ocurrences);

        tokenizer.max_heap.push((10, Reverse(pair_info(b'a', b'b'))));
        tokenizer.max_heap.push((5, Reverse(pair_info(b'c', b'd'))));

        assert_eq!(
            tokenizer.get_winning_token().unwrap(),
            Some(pair_info(b'c', b'd'))
        );
        assert!(!tokenizer.tokens.contains_key(&stale));
    }

    #[test]
    fn get_winning_token_refreshes_a_stale_positive_count() {
        let mut tokenizer = byte_ready_tokenizer();
        let pair = pair_of_bytes(b'a', b'b');
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(0, 3, 0);
        tokenizer.tokens.insert(pair, occurrences);
        tokenizer.max_heap.push((10, Reverse(pair_info(b'a', b'b'))));

        assert_eq!(
            tokenizer.get_winning_token().unwrap(),
            Some(pair_info(b'a', b'b'))
        );
        assert!(tokenizer.max_heap.is_empty());
    }

    #[test]
    fn get_winning_token_breaks_equal_counts_by_lexicographically_lowest_token() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([
            ("ab".to_string(), 1),
            ("ba".to_string(), 1),
        ]));
        tokenizer.initialize().unwrap();

        assert_eq!(
            tokenizer.get_winning_token().unwrap(),
            Some(pair_info(b'a', b'b'))
        );
    }

    #[test]
    fn get_winning_token_skips_heap_entry_without_token_record() {
        let mut tokenizer = byte_ready_tokenizer();
        let valid = pair_of_bytes(b'c', b'd');
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(0, 1, 0);
        tokenizer.tokens.insert(valid, occurrences);
        tokenizer.max_heap.push((2, Reverse(pair_info(b'a', b'b'))));
        tokenizer.max_heap.push((1, Reverse(pair_info(b'c', b'd'))));

        assert_eq!(
            tokenizer.get_winning_token().unwrap(),
            Some(pair_info(b'c', b'd'))
        );
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
    fn next_token_errors_when_winning_pair_cannot_be_resolved() {
        let mut tokenizer = byte_ready_tokenizer();
        // A heap entry whose pair bytes are not registered tokens is skipped
        // as stale, and an empty heap ends training with None.
        tokenizer
            .max_heap
            .push((1, Reverse(TokenInfo::new(b"missing".to_vec(), Some(b"pair".to_vec())))));

        assert_eq!(tokenizer.next_token().unwrap(), None);
        assert!(tokenizer.pair_merges.is_empty());
    }

    #[test]
    fn next_token_returns_none_when_queue_is_empty() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("a".to_string(), 1)]));
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

    #[test]
    fn vocab_size_counts_base_bytes_plus_learned_merges() {
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 1)]));
        assert_eq!(tokenizer.vocab_size(), 0);

        tokenizer.initialize().unwrap();
        assert_eq!(tokenizer.vocab_size(), 256);

        tokenizer.next_token().unwrap().unwrap();
        assert_eq!(tokenizer.vocab_size(), 257);
    }

    #[test]
    fn train_reaches_the_target_vocabulary_size() {
        // "abcdef" (x2) chains into more merges than the target needs.
        let mut tokenizer = BPETokenizer::new(HashMap::from([("abcdef".to_string(), 2)]));
        tokenizer.initialize().unwrap();

        tokenizer.train(260).unwrap();

        assert_eq!(tokenizer.vocab_size(), 260);
    }

    #[test]
    fn train_stops_early_when_the_corpus_is_exhausted() {
        // "ab" can only produce a single merge, so asking for more is fine.
        let mut tokenizer = BPETokenizer::new(HashMap::from([("ab".to_string(), 3)]));
        tokenizer.initialize().unwrap();

        tokenizer.train(100_000).unwrap();

        assert_eq!(tokenizer.vocab_size(), 257);
    }

    #[test]
    fn train_rejects_an_uninitialized_tokenizer() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        assert!(tokenizer.train(300).is_err());
    }

    #[test]
    fn train_and_encode_end_to_end_via_serialization() {
        let mut word_ocurrences = HashMap::new();
        for word in ["the quick brown fox", "the lazy dog", "the fox"] {
            for part in word.split_whitespace() {
                *word_ocurrences.entry(part.to_string()).or_insert(0) += 1;
            }
        }
        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();
        tokenizer.train(270).unwrap();

        let restored = BPETokenizer::from_bytes(&tokenizer.to_bytes().unwrap()).unwrap();
        let text = "the quick brown fox";
        assert_eq!(tokenizer.encode(text), restored.encode(text));
        assert_eq!(
            restored.decode_to_string(&restored.encode(text)).unwrap(),
            text
        );
    }

    #[test]
    fn train_survives_a_winner_with_adjacent_occurrences() {
        // Regression test for a state inconsistency that crashed training with
        // WordNotFound. "ar" wins first and has two ADJACENT occurrences in
        // "ararap" ([0,2) and [2,4)), so the pair between them, (r,a) at byte 1,
        // is inserted twice into the update list (once per neighboring
        // occurrence). An ordered removal loop then mis-syncs and keeps the
        // (r,a) occurrence at byte 3, whose pair the merge destroyed. When (r,a)
        // later wins, that stale occurrence produces a phantom update that
        // deletes the live (a,p) entry for "ararap", and the next winner
        // adjacent to the still-existing (a,p) pair fails with WordNotFound.
        let mut word_ocurrences = HashMap::new();
        word_ocurrences.insert("ararap".to_string(), 1);
        word_ocurrences.insert("ar".to_string(), 10);
        word_ocurrences.insert("ra".to_string(), 2);
        word_ocurrences.insert("ap".to_string(), 1);
        let mut tokenizer = BPETokenizer::new(word_ocurrences);
        tokenizer.initialize().unwrap();

        // Before the fix this failed at the 4th merge (vocab size 260)
        // with `WordNotFound: ararap`.
        let trained = tokenizer.train(1000);
        assert!(trained.is_ok(), "training failed: {:?}", trained.err());
        // The corpus chains into a handful of merges and is then exhausted.
        assert!(tokenizer.vocab_size() < 1000);
        // The victim word still round-trips after training.
        let ids = tokenizer.encode("ararap");
        assert_eq!(tokenizer.decode_to_string(&ids).unwrap(), "ararap");
    }
}
