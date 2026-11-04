use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bpe_tokenizer::{BPETokenizer, PreTokenization};

// Usage:
//   Train:   bpe-tokenizer <corpus-file-or-dir> <model-out> <vocab-size> [--skip-tags]
//   Encode:  bpe-tokenizer --model <model-file> --encode <text>
//   Decode:  bpe-tokenizer --model <model-file> --decode <id> [<id> ...]
//
// Training: <corpus-file-or-dir> is a single text file, or a directory that is
// walked recursively (wikiextractor layout: text/AA/wiki_00, ...).
// --skip-tags drops lines that start with '<' after trimming, which removes
// the wikiextractor <doc ...> and </doc> markers from the word counts.
//
// Inference: encodes @text into token ids (printed space-separated), or
// decodes token ids back into text.

const USAGE: &str = "Usage:
  Train:   bpe-tokenizer <corpus-file-or-dir> <model-out> <vocab-size> [--skip-tags]
  Encode:  bpe-tokenizer --model <model-file> --encode <text>
  Decode:  bpe-tokenizer --model <model-file> --decode <id> [<id> ...]";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();

    // Any invocation containing --model switches to inference mode.
    if let Some(position) = args.iter().position(|arg| arg == "--model") {
        return inference_main(&args, position);
    }
    training_main(&args)
}

fn inference_main(args: &[String], model_position: usize) -> ExitCode {
    let Some(model_path) = args.get(model_position + 1) else {
        eprintln!("--model requires a file path.\n{}", USAGE);
        return ExitCode::from(2);
    };
    let rest = &args[model_position + 2..];
    let result: Result<(String, String), String> =
        if rest.first().map(String::as_str) == Some("--encode") {
            // Everything after --encode is the text to tokenize.
            let text = rest[1..].join(" ");
            encode_with_model(Path::new(model_path), &text).map(|(ids, stats)| {
                let ids_line = ids
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(" ");
                (ids_line, stats)
            })
        } else if rest.first().map(String::as_str) == Some("--decode") {
            let ids: Result<Vec<u32>, String> = rest[1..]
                .iter()
                .map(|arg| {
                    arg.parse::<u32>()
                        .map_err(|_| format!("Not a valid token id: {}", arg))
                })
                .collect();
            match ids {
                Ok(ids) => decode_with_model(Path::new(model_path), &ids)
                    .map(|text| (text, String::new())),
                Err(message) => Err(message),
            }
        } else {
            Err(format!("Missing --encode or --decode mode.\n{}", USAGE))
        };
    match result {
        Ok((output, stats)) => {
            println!("{}", output);
            if !stats.is_empty() {
                eprintln!("{}", stats);
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {}", message);
            ExitCode::from(1)
        }
    }
}

/**
 * Encodes @text with the model at @model_path, returning the token ids and a
 * human-readable stats line.
 */
fn encode_with_model(model_path: &Path, text: &str) -> Result<(Vec<u32>, String), String> {
    let tokenizer = BPETokenizer::load(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;
    let ids = tokenizer.encode(text);
    let stats = format!(
        "{} tokens for {} bytes ({:.1} bytes/token)",
        ids.len(),
        text.len(),
        if ids.is_empty() {
            0.0
        } else {
            text.len() as f64 / ids.len() as f64
        }
    );
    Ok((ids, stats))
}

/**
 * Decodes @ids with the model at @model_path, returning the (lossy) text.
 */
fn decode_with_model(model_path: &Path, ids: &[u32]) -> Result<String, String> {
    let tokenizer = BPETokenizer::load(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;
    tokenizer
        .decode_to_string(ids)
        .map_err(|e| format!("Failed to decode: {}", e))
}

fn training_main(args: &[String]) -> ExitCode {
    // Accept the flag in any position; everything else must be the 3 required arguments.
    let mut positional: Vec<String> = Vec::new();
    let mut skip_tags = false;
    for arg in args {
        match arg.as_str() {
            "--skip-tags" => skip_tags = true,
            _ => positional.push(arg.clone()),
        }
    }
    if positional.len() != 3 {
        eprintln!("{}", USAGE);
        return ExitCode::from(2);
    }
    let corpus_path = PathBuf::from(&positional[0]);
    let model_path = positional[1].clone();
    let vocab_size: usize = match positional[2].parse() {
        Ok(size) if size >= 256 => size,
        _ => {
            eprintln!("Vocab size must be an integer >= 256.");
            return ExitCode::from(2);
        }
    };

    if let Err(message) = run(&corpus_path, &model_path, vocab_size, skip_tags) {
        eprintln!("error: {}", message);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Resolves an output path even when the output file has not been created yet.
fn resolve_output_path(path: &Path) -> Result<PathBuf, String> {
    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|e| format!("Failed to resolve current directory: {}", e))?
            .join(path)
    };

    if let Ok(canonical_path) = fs::canonicalize(&absolute_path) {
        return Ok(canonical_path);
    }

    let file_name = absolute_path
        .file_name()
        .ok_or_else(|| format!("Invalid model output path {}.", path.display()))?;
    let parent = absolute_path.parent().unwrap_or(Path::new("."));
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|e| format!("Failed to access model output directory: {}", e))?;
    Ok(canonical_parent.join(file_name))
}

/**
 * Collects the corpus files to read: @path itself if it is a file,
 * or every file under it (recursively, in sorted order) if it is a directory.
 *
 * Every visited path is resolved to its canonical form and remembered in
 * @visited, so a directory or file that is reachable through multiple paths
 * (including symlinks and directory cycles, e.g. a link back to an ancestor)
 * is walked exactly once. This terminates cycles and prevents counting the
 * same content twice in the word counts.
 */
fn collect_files(
    path: &Path,
    output_path: &Path,
    out: &mut Vec<PathBuf>,
    visited: &mut HashSet<PathBuf>,
) -> Result<(), String> {
    // Following the metadata (fs::metadata, not symlink_metadata) means links
    // resolve to their targets; the @visited set makes that safe.
    let metadata = fs::metadata(path)
        .map_err(|e| format!("Failed to access corpus path {}: {}", path.display(), e))?;

    if metadata.is_dir() {
        let canonical_path = fs::canonicalize(path).map_err(|e| {
            format!("Failed to resolve corpus directory {}: {}", path.display(), e)
        })?;
        if !visited.insert(canonical_path) {
            // Already walked this directory, possibly through a link or cycle.
            return Ok(());
        }
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory {}: {}", path.display(), e))?
            .map(|entry| {
                entry
                    .map(|e| e.path())
                    .map_err(|e| format!("Failed to read directory entry: {}", e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Sorted for deterministic training runs.
        entries.sort();
        for entry in entries {
            collect_files(&entry, output_path, out, visited)?;
        }
    } else if metadata.is_file() {
        let canonical_path = fs::canonicalize(path)
            .map_err(|e| format!("Failed to resolve corpus file {}: {}", path.display(), e))?;
        if canonical_path != output_path && visited.insert(canonical_path.clone()) {
            out.push(canonical_path);
        }
    } else {
        return Err(format!(
            "Corpus path is not a regular file or directory: {}",
            path.display()
        ));
    }
    Ok(())
}

fn run(corpus_path: &Path, model_path: &str, vocab_size: usize, skip_tags: bool) -> Result<(), String> {
    // 1. Resolve the corpus files.
    let output_path = resolve_output_path(Path::new(model_path))?;
    let mut files: Vec<PathBuf> = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    collect_files(corpus_path, &output_path, &mut files, &mut visited)?;
    if files.is_empty() {
        return Err("Corpus path contains no files.".to_string());
    }

    // 2. Stream all files line by line, accumulating pre-token word counts.
    let mut word_ocurrences: HashMap<String, u64> = HashMap::new();
    for file in &files {
        let corpus_file = File::open(file)
            .map_err(|e| format!("Failed to open {}: {}", file.display(), e))?;
        for line in BufReader::new(corpus_file).lines() {
            let line =
                line.map_err(|e| format!("Failed reading {}: {}", file.display(), e))?;
            if skip_tags && line.trim_start().starts_with('<') {
                continue;
            }
            PreTokenization::count_tokens(&line, &mut word_ocurrences);
        }
    }
    let total_ocurrences: u64 = word_ocurrences.values().sum();
    eprintln!(
        "Corpus loaded: {} files, {} unique words, {} total word occurrences",
        files.len(),
        word_ocurrences.len(),
        total_ocurrences
    );

    // 3. Train until the vocabulary reaches the target size.
    let mut tokenizer = BPETokenizer::new(word_ocurrences);
    tokenizer
        .initialize()
        .map_err(|e| format!("Initialization failed: {}", e))?;

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
            Err(e) => {
                return Err(format!(
                    "Training failed at vocab size {}: {}",
                    tokenizer.vocab_size(),
                    e
                ))
            }
        }
    }
    eprintln!("  vocab size: {}", tokenizer.vocab_size());

    // 4. Save the model.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        env::temp_dir().join(format!(
            "bpe-tokenizer-cli-test-{}-{}",
            std::process::id(),
            unique
        ))
    }

    #[test]
    fn collect_files_excludes_the_model_output() {
        let root = temporary_directory();
        fs::create_dir_all(&root).unwrap();
        let corpus_file = root.join("corpus.txt");
        let model_file = root.join("model.bin");
        fs::write(&corpus_file, "corpus").unwrap();
        fs::write(&model_file, "previous model").unwrap();

        let output_path = resolve_output_path(&model_file).unwrap();
        let mut files = Vec::new();
        let mut visited = HashSet::new();
        collect_files(&root, &output_path, &mut files, &mut visited).unwrap();

        assert_eq!(files, vec![fs::canonicalize(corpus_file).unwrap()]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn collect_files_terminates_on_directory_cycles() {
        let root = temporary_directory();
        fs::create_dir_all(&root).unwrap();
        let link = root.join("loop");

        #[cfg(unix)]
        let link_result = std::os::unix::fs::symlink(&root, &link);
        #[cfg(windows)]
        let link_result = std::os::windows::fs::symlink_dir(&root, &link);

        if link_result.is_err() {
            fs::remove_dir_all(root).unwrap();
            return;
        }

        let output_path = resolve_output_path(&root.join("model.bin")).unwrap();
        let mut files = Vec::new();
        let mut visited = HashSet::new();
        collect_files(&root, &output_path, &mut files, &mut visited).unwrap();

        // The cycle resolves to the already visited root: no infinite
        // recursion, no duplicated files.
        assert!(files.is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn collect_files_walks_linked_directories_only_once() {
        let root = temporary_directory();
        let dir = root.join("a");
        fs::create_dir_all(&dir).unwrap();
        let corpus_file = dir.join("corpus.txt");
        fs::write(&corpus_file, "corpus").unwrap();
        let link = root.join("link-to-a");

        #[cfg(unix)]
        let link_result = std::os::unix::fs::symlink(&dir, &link);
        #[cfg(windows)]
        let link_result = std::os::windows::fs::symlink_dir(&dir, &link);

        if link_result.is_err() {
            fs::remove_dir_all(root).unwrap();
            return;
        }

        let output_path = resolve_output_path(&root.join("model.bin")).unwrap();
        let mut files = Vec::new();
        let mut visited = HashSet::new();
        collect_files(&root, &output_path, &mut files, &mut visited).unwrap();

        // The same file is reachable through a/ and link-to-a/ but is
        // collected exactly once.
        assert_eq!(files, vec![fs::canonicalize(&corpus_file).unwrap()]);
        fs::remove_dir_all(root).unwrap();
    }

    /** Trains a small model and writes it to a temporary file. */
    fn temporary_model() -> (PathBuf, HashMap<String, u64>) {
        let mut words = HashMap::new();
        PreTokenization::count_tokens("the quick brown fox", &mut words);
        PreTokenization::count_tokens("the lazy dog", &mut words);
        let mut tokenizer = BPETokenizer::new(words);
        tokenizer.initialize().unwrap();
        tokenizer.train(280).unwrap();

        let model_path = temporary_directory().with_extension("bin");
        tokenizer.save(&model_path).unwrap();
        (model_path, HashMap::new())
    }

    #[test]
    fn encode_with_model_returns_token_ids_and_stats() {
        let (model_path, _) = temporary_model();

        let (ids, stats) = encode_with_model(&model_path, "the quick brown fox").unwrap();

        assert!(!ids.is_empty());
        assert!(stats.contains("bytes/token"));
        // Round trip through the decoder restores the text.
        let text = decode_with_model(&model_path, &ids).unwrap();
        assert_eq!(text, "the quick brown fox");
        fs::remove_file(&model_path).unwrap();
    }

    #[test]
    fn decode_with_model_rejects_unknown_ids() {
        let (model_path, _) = temporary_model();

        assert!(decode_with_model(&model_path, &[999_999]).is_err());
        fs::remove_file(&model_path).unwrap();
    }

    #[test]
    fn encode_with_model_rejects_a_missing_model_file() {
        assert!(encode_with_model(Path::new("missing-model.bin"), "text").is_err());
    }

    #[test]
    fn encode_with_model_handles_empty_text() {
        let (model_path, _) = temporary_model();

        let (ids, _) = encode_with_model(&model_path, "").unwrap();
        assert!(ids.is_empty());
        fs::remove_file(&model_path).unwrap();
    }
}
