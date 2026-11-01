#[cfg(test)]
mod tests {
    use super::*;
    use crate::OcurrencePosition;
    use std::cmp::Reverse;
    use std::collections::HashSet;

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

        let token_ocurrences_cases: Vec<(&str, u64)> = vec![
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
            assert_eq!(
                tokenizer
                    .tokens
                    .get(&temp_token_info)
                    .unwrap()
                    .get_ocurrences_count(),
                expected_count
            );
        }

        let token_position_cases: Vec<(
            &str,
            HashMap<String, Vec<(OcurrencePosition, Option<OcurrencePosition>)>>,
        )> = vec![
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
            let actual_positions = tokenizer
                .tokens
                .get(&temp_token_info)
                .unwrap()
                .get_ocurrences();
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
        let expected_tokens_count =
            (8 + hello_generated_tokens_count + worlod_generated_tokens_count)
                - common_tokens_count;
        assert!(tokenizer.tokens.len() == (expected_tokens_count as usize));

        // Check that new tokens have been added to the max_heap
        assert!(!tokenizer.max_heap.is_empty());

        // Check that the new tokens are present in the tokens map
        for (count, Reverse(token_info)) in &tokenizer.max_heap {
            assert!(tokenizer.tokens.contains_key(token_info));
            assert_eq!(
                tokenizer
                    .tokens
                    .get(token_info)
                    .unwrap()
                    .get_ocurrences_count(),
                *count
            );
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
            tokenizer.tokens[&TokenInfo::new(vec![0xC3], None)].get_ocurrences()["é"][0]
                .0
                .get_byte_offset(),
            &0
        );
        assert_eq!(
            tokenizer.tokens[&TokenInfo::new(vec![0xA9], None)].get_ocurrences()["é"][0]
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
        let position = (
            OcurrencePosition::new(0, 1),
            Some(OcurrencePosition::new(1, 1)),
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
        let position = (
            OcurrencePosition::new(0, 1),
            Some(OcurrencePosition::new(1, 1)),
        );
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
                vec![(
                    OcurrencePosition::new(0, 1),
                    Some(OcurrencePosition::new(1, 1)),
                )],
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
        tokenizer
            .merge_winning_token_and_generate_new_ones(&winning)
            .unwrap();

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
        let positions = vec![(
            OcurrencePosition::new(1, 1),
            Some(OcurrencePosition::new(2, 1)),
        )];

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
            vec![(
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1))
            )]
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
        let positions = vec![(
            OcurrencePosition::new(1, 1),
            Some(OcurrencePosition::new(2, 1)),
        )];

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
            vec![(
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1))
            )]
        );
        assert_eq!(
            map[&adjacent]["abcd"],
            vec![(
                OcurrencePosition::new(2, 1),
                Some(OcurrencePosition::new(3, 1))
            )]
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
        let positions = vec![(
            OcurrencePosition::new(2, 1),
            Some(OcurrencePosition::new(3, 2)),
        )];

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
            vec![(
                OcurrencePosition::new(0, 2),
                Some(OcurrencePosition::new(2, 1))
            )]
        );
        assert_eq!(
            map[&right_adjacent]["aabccd"],
            vec![(
                OcurrencePosition::new(3, 2),
                Some(OcurrencePosition::new(5, 1))
            )]
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
        let positions = vec![(
            OcurrencePosition::new(0, 1),
            Some(OcurrencePosition::new(1, 1)),
        )];

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
            vec![(
                OcurrencePosition::new(1, 1),
                Some(OcurrencePosition::new(2, 1))
            )]
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
        let positions = vec![(
            OcurrencePosition::new(1, 1),
            Some(OcurrencePosition::new(2, 1)),
        )];

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
            vec![(
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1))
            )]
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
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
            (
                OcurrencePosition::new(1, 1),
                Some(OcurrencePosition::new(2, 1)),
            ),
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
                (
                    OcurrencePosition::new(0, 1),
                    Some(OcurrencePosition::new(1, 1))
                ),
                (
                    OcurrencePosition::new(4, 1),
                    Some(OcurrencePosition::new(5, 1))
                ),
            ]
        );
        assert_eq!(
            updates[&right_adjacent]["xabyxaby"],
            vec![
                (
                    OcurrencePosition::new(2, 1),
                    Some(OcurrencePosition::new(3, 1))
                ),
                (
                    OcurrencePosition::new(6, 1),
                    Some(OcurrencePosition::new(7, 1))
                ),
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
        assert!(!tokenizer.tokens[&adjacent]
            .get_ocurrences()
            .contains_key("abc"));
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
        occurrences.add_ocurrence("ab".to_string(), 1, (OcurrencePosition::new(0, 2), None));
        tokenizer.tokens.insert(token.clone(), occurrences);
        let to_remove = HashMap::from([(
            token,
            HashMap::from([("ab".to_string(), vec![(OcurrencePosition::new(0, 2), None)])]),
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
            HashMap::from([("ab".to_string(), vec![(OcurrencePosition::new(0, 2), None)])]),
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

        tokenizer
            .tokens
            .insert(stale.clone(), TokenOcurrences::new());

        let mut valid_ocurrences = TokenOcurrences::new();
        valid_ocurrences.add_ocurrence(
            "cd".to_string(),
            5,
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
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
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
        );
        tokenizer.tokens.insert(token.clone(), occurrences);
        tokenizer.max_heap.push((10, Reverse(token.clone())));

        assert_eq!(tokenizer.get_winning_token().unwrap(), Some(token.clone()));
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
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
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

        assert!(tokenizer
            .token_parts(&TokenInfo::new(b"missing".to_vec(), Some(b"b".to_vec())))
            .is_err());
        assert!(tokenizer
            .token_parts(&TokenInfo::new(b"a".to_vec(), Some(b"missing".to_vec())))
            .is_err());
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
        assert!(tokenizer
            .max_heap
            .iter()
            .any(|(_, Reverse(token))| token == &left_candidate));
        assert!(tokenizer
            .max_heap
            .iter()
            .any(|(_, Reverse(token))| token == &right_candidate));
    }

    #[test]
    fn merge_winning_token_errors_when_word_frequency_is_missing() {
        let mut tokenizer = BPETokenizer::new(HashMap::new());
        let winning = TokenInfo::new(b"a".to_vec(), Some(b"b".to_vec()));
        let mut occurrences = TokenOcurrences::new();
        occurrences.add_ocurrence(
            "ab".to_string(),
            1,
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
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
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
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
            (
                OcurrencePosition::new(0, 1),
                Some(OcurrencePosition::new(1, 1)),
            ),
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
