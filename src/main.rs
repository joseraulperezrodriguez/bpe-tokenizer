use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::ExitCode;

use bpe_tokenizer::{BPETokenizer, PreTokenization};

// Usage: bpe-tokenizer <corpus-file> <model-out> <vocab-size>
//
// Streams the corpus line by line, accumulating pre-token word counts,
// trains the tokenizer until the vocabulary reaches <vocab-size> tokens,
// and writes the model to <model-out>.
//
// Note: the whole word-count map and training state live in memory; this
// driver is functional but will not scale to the full 45 GB corpus until
// the planned interning/memory redesign lands.

const USAGE: &str = "Usage: bpe-tokenizer <corpus-file> <model-out> <vocab-size>";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("{}", USAGE);
        return ExitCode::from(2);
    }
    let corpus_path = &args[1];
    let model_path = &args[2];
    let vocab_size: usize = match args[3].parse() {
        Ok(size) if size >= 256 => size,
        _ => {
            eprintln!("Vocab size must be an integer >= 256.");
            return ExitCode::from(2);
        }
    };

    if let Err(message) = run(corpus_path, model_path, vocab_size) {
        eprintln!("error: {}", message);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

fn run(corpus_path: &str, model_path: &str, vocab_size: usize) -> Result<(), String> {
    // 1. Stream the corpus, accumulating pre-token word counts.
    let mut word_ocurrences: HashMap<String, u64> = HashMap::new();
    let corpus_file =
        File::open(corpus_path).map_err(|e| format!("Failed to open corpus: {}", e))?;
    let reader = BufReader::new(corpus_file);
    for (line_number, line) in reader.lines().enumerate() {
        let line = line
            .map_err(|e| format!("Failed reading corpus line {}: {}", line_number, e))?;
        PreTokenization::count_tokens(&line, &mut word_ocurrences);
    }
    eprintln!(
        "Corpus loaded: {} unique words from {}",
        corpus_path,
        word_ocurrences
            .values()
            .sum::<u64>()
    );

    // 2. Train until the vocabulary reaches the target size.
    let mut tokenizer = BPETokenizer::new(word_ocurrences);
    tokenizer
        .initialize()
        .map_err(|e| format!("Initialization failed: {}", e))?;

    let mut last_reported = tokenizer.vocab_size();
    while tokenizer.vocab_size() < vocab_size {
        match tokenizer.next_token() {
            Ok(Some((token_id, token))) => {
                // Progress report every 1000 learned tokens.
                if token_id % 1000 == 0 {
                    let mut stderr = std::io::stderr();
                    let _ = writeln!(
                        stderr,
                        "  vocab size: {} | latest: {:?}",
                        tokenizer.vocab_size(),
                        String::from_utf8_lossy(&token.to_bytes())
                    );
                    last_reported = tokenizer.vocab_size();
                }
            }
            Ok(None) => {
                eprintln!(
                    "Corpus exhausted at {} tokens (target was {}).",
                    tokenizer.vocab_size(),
                    vocab_size
                );
                break;
            }
            Err(e) => return Err(format!("Training failed at vocab size {}: {}", tokenizer.vocab_size(), e)),
        }
    }
    if tokenizer.vocab_size() == last_reported {
        eprintln!("  vocab size: {}", tokenizer.vocab_size());
    }

    // 3. Save the model.
    tokenizer
        .save(model_path)
        .map_err(|e| format!("Failed to save model: {}", e))?;
    eprintln!(
        "Done: {} tokens written to {} ({} merges, ~{} bytes)",
        tokenizer.vocab_size(),
        model_path,
        tokenizer.vocab_size() - 256,
        8 * (tokenizer.vocab_size() - 256) + 12
    );
    Ok(())
}
