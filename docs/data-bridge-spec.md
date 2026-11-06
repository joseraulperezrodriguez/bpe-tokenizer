# Data Bridge Spec — bulk token export for training (Phase 2/3)

**Purpose:** export the wikiextractor text tree as flat token-id streams that
the Phase 2 model (`gpt-model`) can train on: `train.bin` and `val.bin`.

You implement; this document is the contract and the design decisions.

## File format (fixed — must match `gpt_model/data.py`)

Little-endian, no padding:

```
offset 0:  uint32  n_tokens        # ids that follow in THIS file
offset 4:  uint32  vocab_hint      # vocab size of the model used
offset 8:  n_tokens × uint32 token ids (token id space of the BPE model)
```

Notes:

- uint32, not uint16: vocab can exceed 65,535 (the 100k model does).
- `n_tokens` is not known before streaming: write a zero header first, then
  seek back to offset 0 and rewrite it after counting (`File` + `seek` is
  sufficient; no buffering — a full-corpus export is data too large to hold:
  at ~5.5 bytes/token the 19.4 GB corpus is ~3.5 B tokens ≈ **14 GB of u32
  ids**; ensure the target disk has room twice over if train+val share it).
- `vocab_hint` comes from `tokenizer.vocab_size()` after `load`.

## CLI interface

```
bpe-tokenizer --model <model-file> --export <corpus-dir> <out-dir> [--val-ratio N] [--skip-tags]
```

Fits the existing arg style (`--model` dispatches to inference mode; extend
`inference_main` with an `--export` branch inside `rest`).

- `--val-ratio`: fraction of files routed to `val.bin`, as an integer
  percent (default `10`). Simple and deterministic; no RNG crate needed.
- `--skip-tags`: same semantics as training (drop lines whose trimmed form
  starts with `<`, i.e. wikiextractor `<doc>`/`</doc>` markers). **Pass it by
  default when exporting the wiki corpus** so the training data distribution
  matches what the tokenizer trained on.
- Output files: `<out-dir>/train.bin`, `<out-dir>/val.bin`. Create `out-dir`
  if missing. Refuse to continue if either output already exists unless an
  explicit `--force` flag is given (accidental double-append of a 14 GB file
  is the failure mode this prevents; seek-back header writing means a plain
  overwrite would also silently produce wrong counts if interrupted).

## Determinism and the split

1. Reuse `collect_files` verbatim: canonicalized, sorted, cycle/symlink-safe
   — the same machinery training uses.
2. Split **by file, not by token**, with a Bresenham-style distribution:
   file `i` goes to val iff `floor((i+1)·p/100) > floor(i·p/100)` where
   `p = val_ratio_percent`. Exact ratio at any corpus size, evenly spread
   through the sorted order (no val block from one topic region), no RNG
   crate needed.
3. Rationale for file-level split: token-level splitting leaks near-duplicate
   text between train and val (an article's sentences would straddle both),
   inflating val performance. Keep whole articles on one side.
4. Alternative (rejected for now): hash `canonical_path` to pick the side —
   stable across corpus additions, but adds a hash dependency for little gain
   at this corpus scale. Revisit only if the corpus gets versioned
   incrementally.

## Encoding loop

For each routed file, stream lines with `BufReader::lines()`:

- Skip tag lines when `--skip-tags` (same predicate as training: a *trimmed*
  line starting with `<`).
- Encode `line + "\n"` — `PreTokenization::split` keeps whitespace runs as
  pre-tokens, so appending the newline inside the encoded text is what makes
  article boundaries survive round-trips (`lines()` strips the `\n` — and a
  trailing `\r` on Windows-written files — so this also normalizes line
  endings for free).
- Invalid UTF-8 lines: encode `String::from_utf8_lossy(&bytes)` rather than
  failing the whole export; count occurrences and report at the end.

**Performance — the one real change inside the library:** `encode()` builds a
fresh word cache per call. Per-line calls rebuild it constantly, and
`encode_word` re-derives merges for the hottest words ("the", " of",…) on
every line of 19,404 files. Add a library API that carries the cache across
calls, e.g.:

```rust
pub fn encode_with_cache(&self, text: &str, cache: &mut HashMap<String, Vec<u32>>) -> ...
```

(keyed by owned `String` pre-token; value = encoded ids). `main.rs` holds one
cache for the entire export. Words are heavily repeated, so a single
corpus-wide cache should cut the merge-loop work by orders of magnitude —
measure bytes/sec on one shard first; if per-word encoding is still the
bottleneck, memoizing inside `encode_word` is the next step, not the first.

## Reporting

To stderr, mirroring the trainer's style:

- per-file progress is too chatty at 19k files: report every ~500 files
  (`files done / total tokens / MB read`);
- final summary per file: `train.bin: N tokens (X bytes/token, from F files)`
  and the same for `val.bin`, plus any lossy-UTF-8 line count.

## Acceptance tests (Rust, integration)

Small temp corpus (a few files, a tiny trained model — the existing
`temporary_model()` helper pattern):

1. **Header correctness:** export a 3-file, 2-file val corpus; read the
   headers; `n_tokens` equals the actual id count; `vocab_hint` equals the
   model's vocab size.
2. **Routing:** every input file's text decodes back out of exactly one of
   the two bins.
3. **Round-trip:** decode the entire small `train.bin` and compare against
   the concatenation of its source files' (tag-filtered, `\n`-normalized)
   lines — exact match.
4. **Determinism:** two exports of the same corpus are byte-identical.
5. **Guard rails:** existing output files without `--force` → error; empty
   corpus dir → error.

Python side (`gpt_model/data.py` `TokenDataset`) already has loader tests;
after a real export, run the model's M2 forward on the first batch as the
cross-repo smoke test.

## Documentation to update when implemented

- `bpe-tokenizer/README.md` usage section: add the export line.
- `gpt-model/README.md` data section: the one-command regeneration.
