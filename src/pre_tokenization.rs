use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

#[derive(Clone, Debug, Default)]
pub struct PreTokenization;

// Matches contractions ('s, 't, 're, 've, 'm, 'll, 'd), Unicode letters (\p{L}),
// Unicode numbers (\p{N}), and whitespace (\s: spaces, tabs, newlines, etc.).
// Other characters are grouped, including ASCII punctuation:
// ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \ ] ^ _ ` { | } ~
// A single leading space is included with letter, number, and other-character tokens.
const TOKEN_REGEX: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

static TOKEN_PATTERN: OnceLock<Regex> = OnceLock::new();

impl PreTokenization {
    pub fn count_tokens(text: &str, out: &mut HashMap<String, u32>) {
        TOKEN_PATTERN
            .get_or_init(|| Regex::new(TOKEN_REGEX).expect("token regex must be valid"))
            .find_iter(text)
            .for_each(|matched| {
                let token = matched.as_str();
                let count = out.entry(token.to_string()).or_insert(0);
                *count += 1;
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_tokens(input: &str, tokens: &[&str]) {
        let mut token_map = HashMap::new();
        PreTokenization::count_tokens(input, &mut token_map);

        let mut expected = HashMap::new();
        for token in tokens {
            *expected.entry(token.to_string()).or_insert(0) += 1;
        }
        assert_eq!(token_map, expected, "input: {input:?}");
    }

    #[test]
    fn count_tokens_matches_all_contractions() {
        assert_tokens("'s't're've'm'll'd", &["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]);
    }

    #[test]
    fn count_tokens_matches_unicode_letters_and_numbers() {
        assert_tokens("héllo 猫 Ελληνικά", &["héllo", " 猫", " Ελληνικά"]);
        assert_tokens("123٤٢ Ⅻ", &["123٤٢", " Ⅻ"]);
    }

    #[test]
    fn count_tokens_groups_punctuation_and_symbols() {
        assert_tokens(
            "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
            &["!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"],
        );
        assert_tokens("hello,world!", &["hello", ",", "world", "!"]);
        assert_tokens("hi🙂🚀", &["hi", "🙂🚀"]);
        assert_tokens("'x", &["'", "x"]);
    }

    #[test]
    fn count_tokens_matches_whitespace_runs() {
        assert_tokens("\t\n\r  ", &["\t\n\r  "]);
        assert_tokens("  hello", &["  ", "hello"]);
        assert_tokens(" \tworld", &[" \t", "world"]);
    }

    #[test]
    fn count_tokens_includes_one_leading_space_with_tokens() {
        assert_tokens("hello world 123 !", &["hello", " world", " 123", " !"]);
    }

    #[test]
    fn count_tokens_counts_repeated_matches() {
        assert_tokens("a a a", &["a", " a", " a"]);
        assert_tokens("!!!??? !!!???", &["!!!???", " !!!???"]);
    }

    #[test]
    fn count_tokens_handles_empty_and_whitespace_only_input() {
        assert_tokens("", &[]);
        assert_tokens(" \t\n", &[" \t\n"]);
    }
}