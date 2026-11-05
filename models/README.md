# Models

Trained tokenizer models produced by this crate. All files use the compact
merge-list binary format described in the crate README and load with
`BPETokenizer::load(...)` or `bpe-tokenizer --model <file> --encode <text>`.

## wiki-en-100k.bin

100,000-token byte-level BPE vocabulary trained on the full English Wikipedia
corpus (wikiextractor 3.1.0 plain-text extraction).

| Property | Value |
|---|---|
| Vocabulary size | 100,000 (256 base byte tokens + 99,744 learned merges) |
| Corpus | English Wikipedia, wikiextractor text tree (~19.4 GB, 19,404 files) |
| Word occurrences | 3,806,131,452 total, 10,420,618 unique |
| Training time | ~25 minutes, ~9.7 GB peak RAM |
| Compression (English prose) | ~5.5 bytes/token |
| File size | ~780 KB |

Reproduce with:

```
bpe-tokenizer <wikiextractor-text-dir> models/wiki-en-100k.bin 100000 --skip-tags
```

The `--skip-tags` flag drops the wikiextractor `<doc>`/`</doc>` marker lines
from the word counts. Training is deterministic: the same corpus and arguments
produce a byte-identical model, and ties between equal-count candidates are
broken lexicographically by the pair's bytes.
