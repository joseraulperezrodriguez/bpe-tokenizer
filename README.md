# bpe-tokenizer

### High Level Design
 The BPE Algorithm:
  - We do a pre-tokenization process in which we split the text by symbols and characters to avoid merging tokens beyond punctuation or other symbols.
     - The result of this process is a list of "words" and its ocurrence count. From this point on, an ocurrence of a token, means that it appeared in a word, which we know the actual ocurrence in the corpus so we use that factor to get the global ocurrence count.
  - Initially all tokens are single bytes, so the vocabulary starts with all 256 possible byte values.
  - We keep 2 lists of tokens, the ones that are part of the final vocabulary and the total running tokens, the ones that compounds the corpus and that we continue merging. 
  - For each "running token" for each word where it appears, we form 2 new tokens: merging it to its left and right token.
     - We keep a count of ocurrences of these new tokens and its positions (each ocurrence needs to have information to reconstruct how this token was created, which basically means wich parts of the word were merged to compound the new token)
  - The token with the most number of ocurrences is added to the vocabulary.
    - We remove the occurrences of the parts that formed the new token, if that token reached number of ocurrences 0, we remove it from the "live tokens".
	- We also remove the ocurrences from prev and next, for example if in tokens 'a' 'b' 'c' 'd', we merged 'bc', besides updating 'b' and 'c' ocurrences map, we also need to update 'ab' and 'cd' maps.
  - We repeat this process until the vocabulary contains the desired number of tokens.
 
 Desired Vocabulary size: 100K tokens,
 
 Training Corpus:
  - 45 GB of the English Wikipedia articles, stored in (almost) plain text.
  - Stored locally in my machine.
 
 
 Structures:
  - {token(left, right), [positions]} -> a map where we store where each token appears in the word where it was merged.
     - Left and right are the raw bytes of the tokens that compound this token.
     - Positions and token lengths are measured in bytes within the UTF-8 word.
  - in memory max heap (keyed as (token left, right)) with the running tokens by ocurrences, store only the tokens, and its ocurrence count, so storage is O(running tokens)
  - {word, [tokens]} -> a map where we store which running tokens form the current word.

 
 Process:

  - Add all 256 possible byte values as tokens in the initial vocabulary, with IDs 0 through 255.
  
  - (Executed once) for each word, iterate over its tokens, for each token.

    - create two new pairs (forming the new tokens), the result of merging current token with its previous and next token.

    - update the max heap with this count, using lazy count or a custom heap implementation that does this in place, heapifying the new element.

    - update the tokens map, adding the new pair and where it was formed from.

  - (Repeat until we get the desired number of tokens) Add the top of the heap to the vocabulary list.

    - For each ocurrence of the winning token:
	  - here we need to update the word and tokens map, for that, we iterate over the positions where the winning token is present and do the following:

	    - The positions within the same word are sorted by ocurrence, we can handle the below algorithm accordingly.

        - update the tokens count map for its left and right parts, as well as for the tokens that has left and right as suffix or prefix, removing them from the ocurrences list and possibly from the running tokens if ocurrences reach zero.

           - for example if in tokens 'a' 'b' 'c' 'd', we merged 'bc', besides updating 'b' and 'c' ocurrences map, we also need to update 'ab' and 'cd' maps.
		    - this update means going to the positions that are part of the winning token, find the tokens that involve them (constructive algorithm) and remove this occurence from their possitions list.
	    - update the word map, by merging the winning tokens in that word token list.
	  - We need special attention to overlapping merges. If 2 winning tokens overlap, for example in "aaa", there is "aa" and "aa" adjacent tokens to merge, the merge process avoid overlapping by jumping by 2 elements after it merges 2 tokens.
  
 Other things to consider:
   - Tie break rule: ties are broken lexicographically (lowest pair first).
   - The count of a pair equals the total length of its positions lists (weighted by word occurrence counts) — no separate claim computation.

### Implementation Design

The trainer represents token contents as byte vectors. Pre-tokenized words remain UTF-8 strings as map keys, but each word is split with `str::bytes()` into one-byte initial tokens. The base vocabulary is initialized in byte-value order, and learned token IDs start at 256. Position offsets and token lengths use byte units. `TokenInfo::to_bytes()` returns the exact token bytes; its string conversion is lossy for invalid UTF-8 and should only be used for display.

Encoding/decoding: each learned merge is stored in a `(left ID, right ID) -> merged ID` table. `encode` pre-tokenizes the text, then per word repeatedly merges the adjacent pair with the lowest learned ID (the earliest-learned merge wins, ties resolved left to right) until no learned pair remains. `decode` concatenates the bytes of each token ID; `decode_to_string` is lossy for non-UTF-8 boundaries.

Serialization (`to_bytes`/`from_bytes`, `save`/`load`): the model file stores only the learned merge list, in merged-ID order, in a compact little-endian binary format (`BPET` magic, version, merge count, then `left_id, right_id` per merge). Token bytes, the ID maps, and the base byte vocabulary are all rebuilt deterministically from that list on load, so a 100K-merge model is roughly 800 KB on disk. A loaded tokenizer is ready for encode/decode with no training state.

Training driver: `train(target_vocab_size)` runs merges until the vocabulary reaches the target (256 base tokens + learned merges) or the corpus is exhausted; `vocab_size()` reports the current size and is what an LLM's embedding table must be sized to. The `bpe-tokenizer` binary streams a corpus file line by line, accumulates word counts, trains with progress reporting, and writes the model: `bpe-tokenizer <corpus-file> <model-out> <vocab-size>`.
