#[derive(Clone, Debug, Default, Eq, Hash)]
pub struct TokenInfo {
    left: Vec<u8>,
    right: Option<Vec<u8>>,
}

impl TokenInfo {
    pub fn new(left: Vec<u8>, right: Option<Vec<u8>>) -> Self {
        Self { left, right }
    }

    pub fn get_left(&self) -> &[u8] {
        &self.left
    }

    pub fn get_right(&self) -> Option<&[u8]> {
        self.right.as_deref()
    }

    pub fn to_string(&self) -> String {
        String::from_utf8_lossy(&self.to_bytes()).into_owned()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = self.left.clone();
        if let Some(right) = &self.right {
            bytes.extend_from_slice(right);
        }
        bytes
    }

    pub fn size(&self) -> usize {
        self.left.len() + self.right.as_ref().map_or(0, Vec::len)
    }

    pub fn merge(left: &TokenInfo, right: &TokenInfo) -> TokenInfo {
        TokenInfo::new(left.to_bytes(), Some(right.to_bytes()))
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
        self.left
            .cmp(&other.left)
            .then_with(|| self.right.cmp(&other.right))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Token {
    token_id: u32,
    left_part: u32,
    right_part: Option<u32>,
}

impl Token {

    pub fn new(token_id: u32, left_part: u32, right_part: Option<u32>) -> Self {
       Self{
        token_id, 
        left_part,
        right_part,
       }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_info_to_string() {
        let token_info = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
        assert_eq!(token_info.to_string(), "helloworld");
        assert_eq!(token_info.to_bytes(), b"helloworld");
    }

    #[test]
    fn test_token_info_size() {
        let token_info = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
        assert_eq!(token_info.size(), 10);
    }

    #[test]
    fn test_token_info_merge() {
        let left = TokenInfo::new(b"hello".to_vec(), None);
        let right = TokenInfo::new(b"world".to_vec(), None);
        let merged = TokenInfo::merge(&left, &right);
        assert_eq!(merged.to_string(), "helloworld");
        assert_eq!(merged.to_bytes(), b"helloworld");
    }

    #[test]
    fn test_token_info_ordering() {
        {
          let token_info1 = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
          let token_info2 = TokenInfo::new(b"hello".to_vec(), Some(b"rust".to_vec()));
          assert!(token_info1 > token_info2);
        }
        {
            let token_info1 = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
            let token_info2 = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
            assert_eq!(token_info1, token_info2);
        }
        {
            let token_info1 = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
            let token_info2 = TokenInfo::new(b"hello".to_vec(), None);
            assert!(token_info1 > token_info2);
        }
        {
            let token_info1 = TokenInfo::new(b"hello".to_vec(), None);
            let token_info2 = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
            assert!(token_info1 < token_info2);
        }
        {
            let token_info1 = TokenInfo::new(b"hello".to_vec(), None);
            let token_info2 = TokenInfo::new(b"hello".to_vec(), None);
            assert_eq!(token_info1, token_info2);
        }
        {
            let token_info1 = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
            let token_info2 = TokenInfo::new(b"hello".to_vec(), Some(b"world".to_vec()));
            assert_eq!(token_info1, token_info2);
        }
    }

    #[test]
    fn token_info_preserves_non_utf8_bytes() {
        let token_info = TokenInfo::new(vec![0xC3], Some(vec![0xA9]));

        assert_eq!(token_info.to_bytes(), [0xC3, 0xA9]);
        assert_eq!(token_info.size(), 2);
    }
}