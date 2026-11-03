use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bpe_tokenizer::{BPETokenizer, PreTokenization};

// Usage: bpe-tokenizer <corpus-file-or-dir> <model-out> <vocab-size> [--skip-tags]
//
// <corpus-file-or-dir> is a single text file, or a directory that is walked
// recursively (wikiextractor layout: text/AA/wiki_00, text/AA/wiki_01, ...).
// --skip-tags drops lines that start with '<' after trimming, which removes
// the wikiextractor <doc ...> and </doc> markers from the word counts.
//
// Note: the whole word-count map and training state live in memory; this
// driver is functional but will not scale to the full corpus until the
// planned interning/memory redesign lands.

const USAGE: &str =
    "Usage: bpe-tokenizer <corpus-file-or-dir> <model-out> <vocab-size> [--skip-tags]";

fn main() -> ExitCode {
    // Accept the flag in any position; everything else must be the 3 required arguments.
    let mut positional: Vec<String> = Vec::new();
    let mut skip_tags = false;
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--skip-tags" => skip_tags = true,
            _ => positional.push(arg),
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
}
