
use bpe_tokenizer::PreTokenization;
use std::collections::HashMap;

fn main() {
    let mut token_map = HashMap::new();
    PreTokenization::count_tokens("hello world", &mut token_map);
    println!("token map: {:?}", token_map);
}
