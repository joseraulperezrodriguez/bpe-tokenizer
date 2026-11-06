use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use bpe_tokenizer::{BPETokenizer, PreTokenization};

// Usage:
//   Train:   bpe-tokenizer <corpus-file-or-dir> <model-out> <vocab-size> [--skip-tags]
//   Encode:  bpe-tokenizer --model <model-file> --encode <text>
//   Decode:  bpe-tokenizer --model <model-file> --decode <id> [<id> ...]
//   Export:  bpe-tokenizer --model <model-file> --export <corpus-dir> <out-dir> [--val-ratio <pct>] [--skip-tags] [--force]
//
// Training: <corpus-file-or-dir> is a single text file, or a directory that is
// walked recursively (wikiextractor layout: text/AA/wiki_00, ...).
// --skip-tags drops lines that start with '<' after trimming, which removes
// the wikiextractor <doc ...> and </doc> markers from the word counts.
//
// Inference: encodes @text into token ids (printed space-separated), or
// decodes token ids back into text.
//
// Export: walks the corpus and writes two flat token-id files for training,
// <out-dir>/train.bin and <out-dir>/val.bin (format: docs/data-bridge-spec.md).
// Files are routed whole to train or val, spread evenly in sorted order.
// The format is little-endian uint32: header [n_tokens, vocab_hint], then the ids.

const USAGE: &str = "Usage:
  Train:   bpe-tokenizer <corpus-file-or-dir> <model-out> <vocab-size> [--skip-tags]
  Encode:  bpe-tokenizer --model <model-file> --encode <text>
  Decode:  bpe-tokenizer --model <model-file> --decode <id> [<id> ...]
  Export:  bpe-tokenizer --model <model-file> --export <corpus-dir> <out-dir> [--val-ratio <pct>] [--skip-tags] [--force]";

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
    let result: Result<(), String> = if rest.first().map(String::as_str) == Some("--export") {
        export_main(Path::new(model_path), &rest[1..])
    } else if rest.first().map(String::as_str) == Some("--encode") {
        let text = rest[1..].join(" ");
        encode_with_model(Path::new(model_path), &text).map(|(ids, stats)| {
            println!(
                "{}",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            );
            if !stats.is_empty() {
                eprintln!("{}", stats);
            }
        })
    } else if rest.first().map(String::as_str) == Some("--decode") {
        let ids: Result<Vec<u32>, String> = rest[1..]
            .iter()
            .map(|arg| {
                arg.parse::<u32>()
                    .map_err(|_| format!("Not a valid token id: {}", arg))
            })
            .collect();
        ids.and_then(|ids| decode_with_model(Path::new(model_path), &ids))
            .map(|text| println!("{}", text))
    } else {
        Err(format!("Missing --encode, --decode or --export mode.\n{}", USAGE))
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
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

//---------------------------------------------------------------------------

/**
 * Streaming writer for one token .bin file. The 8-byte header cannot be known
 * before streaming, so a zero header is written up front and rewritten by
 * seek-back in finalize().
 */
struct ExportWriter {
    writer: BufWriter<File>,
    path: PathBuf,
    tokens: u64,
    input_bytes: u64,
}

impl ExportWriter {
    fn create(path: &Path) -> Result<Self, String> {
        let file =
            File::create(path).map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        writer
            .write_all(&[0u8; 8])
            .map_err(|e| format!("Failed to write {}: {}", path.display(), e))?;
        Ok(Self {
            writer,
            path: path.to_path_buf(),
            tokens: 0,
            input_bytes: 0,
        })
    }

    fn write_ids(&mut self, ids: &[u32], input_bytes: u64) -> Result<(), String> {
        for id in ids {
            self.writer
                .write_all(&id.to_le_bytes())
                .map_err(|e| format!("Failed to write {}: {}", self.path.display(), e))?;
        }
        self.tokens += ids.len() as u64;
        self.input_bytes += input_bytes;
        Ok(())
    }

    /** Finishes a token file: flush, seek back to the header, write the counts. */
    fn finalize(self, vocab_hint: u32) -> Result<(u64, u64), String> {
        if self.tokens > u32::MAX as u64 {
            return Err(format!(
                "{} holds more than u32::MAX tokens; the format cannot describe it.",
                self.path.display()
            ));
        }
        let mut file = self
            .writer
            .into_inner()
            .map_err(|e| format!("Failed to flush {}: {}", self.path.display(), e))?;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("Failed to seek {}: {}", self.path.display(), e))?;
        // The header stores u32 values: tokens fit (guarded above), and the
        // u64→u32 cast is what keeps the header exactly 8 bytes wide.
        let n_tokens = u32::try_from(self.tokens).expect("guarded above");
        file.write_all(&n_tokens.to_le_bytes())
            .and_then(|_| file.write_all(&vocab_hint.to_le_bytes()))
            .and_then(|_| file.sync_all())
            .map_err(|e| format!("Failed to finalize {}: {}", self.path.display(), e))?;
        Ok((self.tokens, self.input_bytes))
    }
}

/**
 * Parses --export arguments: <corpus-dir> <out-dir> [--val-ratio <pct>]
 * [--skip-tags] [--force].
 *
 * File i routes to val when floor((i+1) * val_ratio / 100) differs from
 * floor(i * val_ratio / 100) — a Bresenham-style distribution that spreads
 * val evenly through the sorted corpus order at exactly the requested
 * ratio, for any corpus size (10% of 19,404 files is ~1,940 files; 25% of a
 * 4-file test corpus is exactly 1 file, at a different index each).
 */
fn export_main(
    model_path: &Path,
    args: &[String],
) -> Result<(), String> {
    let mut positional: Vec<String> = Vec::new();
    let mut val_ratio_percent: usize = 10;
    let mut skip_tags = false;
    let mut force = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--val-ratio" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| "--val-ratio requires a value (0-100).".to_string())?;
                val_ratio_percent = value
                    .parse()
                    .map_err(|_| format!("Not a valid --val-ratio: {}", value))?;
                if val_ratio_percent > 100 {
                    return Err("--val-ratio must be between 0 and 100.".to_string());
                }
            }
            "--skip-tags" => skip_tags = true,
            "--force" => force = true,
            other => positional.push(other.to_string()),
        }
        index += 1;
    }
    if positional.len() != 2 {
        return Err(format!(
            "--export requires <corpus-dir> and <out-dir>.\n{}",
            USAGE
        ));
    }
    let corpus_path = PathBuf::from(&positional[0]);
    let out_dir = PathBuf::from(&positional[1]);
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("Failed to create {}: {}", out_dir.display(), e))?;
    let train_path = out_dir.join("train.bin");
    let val_path = out_dir.join("val.bin");
    for output in [train_path.as_path(), val_path.as_path()] {
        if output.exists() && !force {
            return Err(format!(
                "{} already exists; pass --force to overwrite.",
                output.display()
            ));
        }
    }
    export_corpus(
        model_path,
        &corpus_path,
        &out_dir,
        val_ratio_percent,
        skip_tags,
    )
}

/**
 * Walks @corpus_path and streams every line's token ids into train.bin /
 * val.bin under @out_dir, then finishes both headers and reports.
 */
fn export_corpus(
    model_path: &Path,
    corpus_path: &Path,
    out_dir: &Path,
    val_ratio_percent: usize,
    skip_tags: bool,
) -> Result<(), String> {
    let tokenizer = BPETokenizer::load(model_path)
        .map_err(|e| format!("Failed to load model: {}", e))?;
    let vocab_hint = tokenizer.vocab_size() as u32;

    // Files are routed whole (not token-by-token) so near-duplicate text
    // cannot straddle the train/val boundary.
    let mut files: Vec<PathBuf> = Vec::new();
    let mut visited: HashSet<PathBuf> = HashSet::new();
    let model_output = resolve_output_path(model_path)?;
    collect_files(corpus_path, &model_output, &mut files, &mut visited)?;
    // A rerun with --force must not encode the previous run's own output.
    let canonical_out_dir = fs::canonicalize(out_dir).map_err(|e| {
        format!("Failed to resolve {}: {}", out_dir.display(), e)
    })?;
    files.retain(|file| !file.starts_with(&canonical_out_dir));
    let total_files = files.len();
    if total_files == 0 {
        return Err("Corpus path contains no files.".to_string());
    }

    let mut train_writer = ExportWriter::create(&out_dir.join("train.bin"))?;
    let mut val_writer = ExportWriter::create(&out_dir.join("val.bin"))?;
    // One cache shared across the entire corpus: repeated words are encoded once.
    let mut word_cache: HashMap<String, Vec<u32>> = HashMap::new();
    let mut lossy_lines: u64 = 0;
    let mut train_files: usize = 0;
    let mut val_files: usize = 0;

    for (index, file) in files.iter().enumerate() {
        let writer = if ((index + 1) * val_ratio_percent) / 100 > (index * val_ratio_percent) / 100 {
            val_files += 1;
            &mut val_writer
        } else {
            train_files += 1;
            &mut train_writer
        };
        let corpus_file =
            File::open(file).map_err(|e| format!("Failed to open {}: {}", file.display(), e))?;
        let mut reader = BufReader::new(corpus_file);
        let mut buffer: Vec<u8> = Vec::new();
        loop {
            buffer.clear();
            let read = reader
                .read_until(b'\n', &mut buffer)
                .map_err(|e| format!("Failed reading {}: {}", file.display(), e))?;
            if read == 0 {
                break;
            }
            // Strip the trailing newline (and a \r before it), then re-add the
            // '\n' inside the encoded text: line endings normalize for free and
            // article boundaries survive round-trips.
            let mut line_bytes: &[u8] = &buffer;
            if line_bytes.last() == Some(&b'\n') {
                line_bytes = &line_bytes[..line_bytes.len() - 1];
            }
            if line_bytes.last() == Some(&b'\r') {
                line_bytes = &line_bytes[..line_bytes.len() - 1];
            }
            let line = match std::str::from_utf8(line_bytes) {
                Ok(line) => line.to_string(),
                Err(_) => {
                    // Invalid UTF-8 does not abort a multi-hour export.
                    lossy_lines += 1;
                    String::from_utf8_lossy(line_bytes).into_owned()
                }
            };
            if skip_tags && line.trim_start().starts_with('<') {
                continue;
            }
            let mut input = String::with_capacity(line.len() + 1);
            input.push_str(&line);
            input.push('\n');
            let input_bytes = input.len() as u64;
            let ids = tokenizer.encode_with_cache(&input, &mut word_cache);
            writer.write_ids(&ids, input_bytes)?;
        }
        if (index + 1) % 500 == 0 || index + 1 == total_files {
            eprintln!(
                "  {}/{} files | train: {} tokens | val: {} tokens",
                index + 1,
                total_files,
                train_writer.tokens,
                val_writer.tokens
            );
        }
    }

    let (train_tokens, train_bytes) = train_writer.finalize(vocab_hint)?;
    let (val_tokens, val_bytes) = val_writer.finalize(vocab_hint)?;

    let report = |name: &str, tokens: u64, bytes: u64, file_count: usize| {
        let bytes_per_token = if tokens > 0 {
            bytes as f64 / tokens as f64
        } else {
            0.0
        };
        eprintln!(
            "{}: {} tokens from {} files ({:.1} bytes/token)",
            name, tokens, file_count, bytes_per_token
        );
    };
    report("train.bin", train_tokens, train_bytes, train_files);
    report("val.bin", val_tokens, val_bytes, val_files);
    if lossy_lines > 0 {
        eprintln!("warning: {} lines contained invalid UTF-8 (encoded lossily)", lossy_lines);
    }
    Ok(())
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

    //-----------------------------------------------------------------------
    // --export tests

    /** Reads a token .bin file: (n_tokens header, vocab header, ids). */
    fn read_export_file(path: &Path) -> (u32, u32, Vec<u32>) {
        let bytes = fs::read(path).unwrap();
        assert!(bytes.len() >= 8, "missing header");
        let n_tokens = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let vocab_hint = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let ids: Vec<u32> = bytes[8..]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(ids.len() as u32, n_tokens, "header count must match the body");
        (n_tokens, vocab_hint, ids)
    }

    /** Creates a corpus with @files as (file_stem, content) pairs. */
    fn export_test_corpus(root: &Path, files: &[(&str, &str)]) {
        fs::create_dir_all(root).unwrap();
        for (name, content) in files {
            fs::write(root.join(name), content).unwrap();
        }
    }

    /**
     * Exports @corpus with a trained model and returns the decoded text of
     * (train, val) plus the recorded vocab hint.
     */
    fn export_and_decode(
        model_path: &Path,
        corpus: &Path,
        out_dir: &Path,
        args: &[&str],
    ) -> (String, String, u32) {
        let mut all_args: Vec<String> = vec![
            corpus.to_string_lossy().into_owned(),
            out_dir.to_string_lossy().into_owned(),
        ];
        for arg in args {
            all_args.push((*arg).to_string());
        }
        export_main(model_path, &all_args).unwrap();

        let tokenizer = BPETokenizer::load(model_path).unwrap();
        let (train_n, train_vocab, train_ids) = read_export_file(&out_dir.join("train.bin"));
        let (val_n, val_vocab, val_ids) = read_export_file(&out_dir.join("val.bin"));
        assert_eq!(train_vocab, val_vocab);
        let train = tokenizer.decode_to_string(&train_ids).unwrap();
        let val = tokenizer.decode_to_string(&val_ids).unwrap();
        assert_eq!(train_ids.len() as u32, train_n);
        assert_eq!(val_ids.len() as u32, val_n);
        (train, val, val_vocab)
    }

    #[test]
    fn export_writes_valid_headers_and_routes_whole_files() {
        let (model_path, _) = temporary_model();
        let root = temporary_directory();
        let corpus = root.join("corpus");
        export_test_corpus(
            &corpus,
            &[
                ("f0", "trainmarker alpha\n"),
                ("f1", "trainmarker beta\n"),
                ("f2", "trainmarker gamma\n"),
                ("f3", "valmarker zero\nvalmarker one\n"),
            ],
        );
        let out_dir = root.join("data");

        // 25% of 4 files is exactly 1 file; the Bresenham routing picks index 3.
        let (train, val, vocab_hint) = export_and_decode(
            &model_path,
            &corpus,
            &out_dir,
            &["--val-ratio", "25"],
        );
        let expected_vocab = BPETokenizer::load(&model_path).unwrap().vocab_size() as u32;
        assert_eq!(vocab_hint, expected_vocab);

        // Whole files are routed: val is exactly f3, train exactly the rest.
        assert_eq!(val, "valmarker zero\nvalmarker one\n");
        assert_eq!(
            train,
            "trainmarker alpha\ntrainmarker beta\ntrainmarker gamma\n"
        );
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(&model_path).unwrap();
    }

    #[test]
    fn export_is_deterministic() {
        let (model_path, _) = temporary_model();
        let root = temporary_directory();
        let corpus = root.join("corpus");
        export_test_corpus(
            &corpus,
            &[("a", "alpha\n"), ("b", "beta\n"), ("c", "gamma\n")],
        );

        let out_a = root.join("data-a");
        let out_b = root.join("data-b");
        for out in [&out_a, &out_b] {
            let args = vec![
                corpus.to_string_lossy().into_owned(),
                out.to_string_lossy().into_owned(),
            ];
            export_main(&model_path, &args).unwrap();
        }

        assert_eq!(
            fs::read(out_a.join("train.bin")).unwrap(),
            fs::read(out_b.join("train.bin")).unwrap()
        );
        assert_eq!(
            fs::read(out_a.join("val.bin")).unwrap(),
            fs::read(out_b.join("val.bin")).unwrap()
        );
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(&model_path).unwrap();
    }

    #[test]
    fn export_rejects_existing_outputs_without_force() {
        let (model_path, _) = temporary_model();
        let root = temporary_directory();
        let corpus = root.join("corpus");
        export_test_corpus(&corpus, &[("a", "alpha\n")]);
        let out_dir = root.join("data");
        fs::create_dir_all(&out_dir).unwrap();
        fs::write(out_dir.join("train.bin"), "previous").unwrap();

        let args = vec![
            corpus.to_string_lossy().into_owned(),
            out_dir.to_string_lossy().into_owned(),
        ];
        let err = export_main(&model_path, &args).unwrap_err();
        assert!(err.contains("--force"));

        // With --force the previous output is replaced, not appended to.
        let mut forced = args;
        forced.push("--force".to_string());
        export_main(&model_path, &forced).unwrap();
        let (n_tokens, _, ids) = read_export_file(&out_dir.join("train.bin"));
        assert_eq!(n_tokens as usize, ids.len());
        assert!(fs::read(&out_dir.join("train.bin")).unwrap().len() > 8);
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(&model_path).unwrap();
    }

    #[test]
    fn export_skips_tag_lines_and_normalizes_line_endings() {
        let (model_path, _) = temporary_model();
        let root = temporary_directory();
        let corpus = root.join("corpus");
        // CRLF endings and wikiextractor-style markers; only content remains.
        export_test_corpus(
            &corpus,
            &[("a", "<doc id=\"1\">\r\nreal content here\r\n</doc>\r\n")],
        );
        let out_dir = root.join("data");

        let (train, _, _) = export_and_decode(&model_path, &corpus, &out_dir, &["--skip-tags"]);
        assert_eq!(train, "real content here\n");
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(&model_path).unwrap();
    }

    #[test]
    fn export_never_encodes_its_own_output_directory() {
        let (model_path, _) = temporary_model();
        let root = temporary_directory();
        let corpus = root.join("corpus");
        export_test_corpus(&corpus, &[("a", "alpha\n")]);
        let out_dir = corpus.join("data"); // nested inside the corpus
        fs::create_dir_all(&out_dir).unwrap();
        fs::write(out_dir.join("stale.bin"), "stale output from a previous run\n").unwrap();

        let (train, _, _) = export_and_decode(&model_path, &corpus, &out_dir, &[]);
        assert_eq!(train, "alpha\n");
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(&model_path).unwrap();
    }

    #[test]
    fn export_rejects_an_empty_corpus_and_bad_arguments() {
        let (model_path, _) = temporary_model();
        let root = temporary_directory();
        let corpus = root.join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        let out_dir = root.join("data");

        let err = export_main(
            &model_path,
            &[
                corpus.to_string_lossy().into_owned(),
                out_dir.to_string_lossy().into_owned(),
                "--val-ratio".to_string(),
                "101".to_string(),
            ],
        )
        .unwrap_err();
        assert!(err.contains("--val-ratio"));

        let err = export_main(&model_path, &[]).unwrap_err();
        assert!(err.contains("--export requires"));

        fs::remove_dir_all(root).unwrap();
        fs::remove_file(&model_path).unwrap();
    }
}
