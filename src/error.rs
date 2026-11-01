use std::error::Error;
use std::fmt::{Display, Formatter, Result};

#[derive(Debug)]
pub enum TokenizationError {
    MissingToken(String),
    WordNotFound(String),
    UnknownTokenId(u32),
    UnexpectedError(String),
}

impl Display for TokenizationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            TokenizationError::MissingToken(token) => write!(f, "Missing token: {}", token),
            TokenizationError::WordNotFound(word) => write!(f, "Word not found: {}", word),
            TokenizationError::UnknownTokenId(id) => write!(f, "Unknown token id: {}", id),
            TokenizationError::UnexpectedError(message) => write!(f, "{}", message),
        }
    }
}

impl Error for TokenizationError {}
