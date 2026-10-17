use std::collections::HashMap;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;
use std::error::Error;


#[derive(Clone, Debug, Default, Eq, Hash)]
pub struct TokenInfo {
    left: String,
    right: Option<String>,
}

impl TokenInfo {
    pub fn new(left: String, right: Option<String>) -> Self {
        Self { left, right }
    }
    pub fn to_string(&self) -> String {
        match &self.right {
            Some(right) => format!("{}{}", self.left, right),
            None => format!("{}", self.left),
        }
    }

    pub fn size(&self) -> u32 {
        match &self.right {
            Some(right) => self.left.len() as u32 + right.len() as u32,
            None => self.left.len() as u32,
        }
    }

    pub fn merge(left: &TokenInfo, right: &TokenInfo) -> TokenInfo {
        let left_str = left.to_string();
        let right_str = right.to_string();
        TokenInfo::new(left_str, Some(right_str))
    }
}

impl PartialEq for TokenInfo {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.right == other.right
    }
}

impl PartialOrd for TokenInfo {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TokenInfo {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.left == other.left {
            match (&self.right, &other.right) {
                (Some(self_right), Some(other_right)) => self_right.cmp(other_right),
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        } else {
            self.left.cmp(&other.left)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Token {
    token_id: u32,
    token_info: TokenInfo,
}

#[derive(Clone, Debug, Default)]
pub struct TokenOcurrences {
    tokens_word_ocurrences: HashMap<String, Vec<((u32, u32), Option<(u32, u32)>)>>,
    count_ocurrences: u32,
}

impl TokenOcurrences {
   pub fn new() -> Self {
        Self {
            tokens_word_ocurrences: HashMap::new(),
            count_ocurrences: 0,
        }
    }

   pub fn add_ocurrence(&mut self, token: String, token_ocurrencies: u32, position: ((u32, u32), Option<(u32, u32)>)) {
        self.tokens_word_ocurrences
            .entry(token)
            .or_insert_with(Vec::new)
            .push(position);
        self.count_ocurrences += token_ocurrencies;
    }

   pub fn get_ocurrences_count(&self) -> u32 {
        self.count_ocurrences
    }

   pub fn get_ocurrences(&self) -> &HashMap<String, Vec<((u32, u32), Option<(u32, u32)>)>> {
        &self.tokens_word_ocurrences
    } 

}

#[derive(Debug)]
pub enum TokenizationError {
    MissingToken(String),
    WordNotFound(String),
}

impl Display for TokenizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            TokenizationError::MissingToken(token) => write!(f, "Missing token: {}", token),
            TokenizationError::WordNotFound(word) => write!(f, "Word not found: {}", word),
        }
    }
}

impl Error for TokenizationError {}