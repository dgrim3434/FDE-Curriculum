# Week 9: pre-registered experiment plan

Written and committed before any retrieval result exists. Every number below comes from
`results/bounds/bounds.json` (measured) or is a stated choice with its reason.

## 1. Corpora and splits
- **Regime A**: SciFact, BEIR test split (300 queries), document-level relevance, nDCG@10.
- **Regime B**: Qasper test split (416 papers), evidence located as character spans,
  evidence recall within a token budget, searched within each question's own paper.
- **Tuning data**: SciFact train queries and Qasper dev only. Nothing is tuned on a test split.

## 2. Held constant in every configuration
- Retrieval embedder: sentence-transformers/all-MiniLM-L6-v2, max_seq_length 256, L2-normalized.
- Exact search; Regime A aggregation MaxP; k = 10; linear gain; stable tie-breaking.
- Regime B primary budget B = 512 embedder tokens (secondary: 256, 1024). At B = 512, the grid's
  sizes fit between 2 (size 254) and 8 (size 64) chunks.
- Document text: SciFact body text (title handled only by the prefix block); Qasper rendered by
  `render_paper`. Sentence spans from en_core_web_sm, cached once per corpus.
- Bootstrap: 10,000 resamples, seed 0. Regime B uses the cluster bootstrap (resampling papers).
- Coherence: representation fit, measured with all-mpnet-base-v2 (a different model from the one
  that places semantic cuts); random-chunk baseline n = 300, seed 0, once per corpus.

## 3. Bounds (from bounds.json)
| bound | value | evidence |
|---|---|---|
| hard ceiling | 254 content tokens (256 − 2 specials) | read + computed |
| ceiling with title prefix | 230 (254 − p95 title prefix of 24) | measured + computed |
| floor anchor | p90 sentence length: 58 (SciFact), 52 (Qasper) | measured |
| budget (4,096 window, r = 1 assumed) | k_max 38 / 27 / 21 / 15 / 11 at sizes 64 / 96 / 128 / 192 / 254: not binding | computed |
| whole-document truncation | SciFact 63% of documents, 26% of tokens; Qasper 100% / 95% | measured |
| structure (has a blank line) | SciFact 0.0, Qasper 1.0 | measured |

## 4. Grid (one factor varies per block)
**Regime A (SciFact)**
- A0 reference: whole body text, no chunking (truncated by the model).
- A1 size: fixed-size, overlap 0, at 64, 96, 128, 192, 254.
- A2 strategy at 128: fixed vs sentence vs semantic. Recursive is omitted: structure is 0.0, so it
  produces exactly the sentence chunker's output on this corpus.

**Regime B (Qasper)**
- B1 size: fixed-size, overlap 0, at 64, 96, 128, 192, 254.
- B2 strategy at 128: fixed, sentence (overlap 0), recursive (merge within paragraphs,
  min_tokens 52), semantic (window 2, min_tokens 52, percentile size-matched; see section 6).
- B3 overlap: fixed-size at the size chosen by rule 5a, overlap 0 vs 52 (Qasper's p90 sentence).
- B4 prefix: the chosen configuration with embed mode "none" vs "title". If the chosen size is
  above 230, the prefix run uses size 230 (the prefix ceiling).

Every configuration also runs the chunk analyzer (size, boundaries, overlap, coherence).

## 5. Decision rules
a. **Size** (per corpus): ship the smallest size in A1/B1 whose paired-difference 95% CI against the
   best size includes zero, provided it passes the floor audit (rule d).
b. **Primary comparisons** (confirmatory): on Qasper at size 128, evidence recall@512:
   sentence vs fixed, recursive vs fixed, semantic vs fixed. 3 comparisons, Bonferroni level
   1 − 0.05/3 = 0.983. Everything else is reported as exploratory.
c. **Reading a CI**: tie margin 0.01 for nDCG@10 and 0.02 for evidence recall, chosen as smaller
   than any difference that would change what I'd ship. A CI above 0 = better; below 0 = worse;
   containing 0 and within the margin = tie; containing 0 and wider than the margin = inconclusive.
   The minimum detectable effect is reported next to every tie.
d. **Floor audit**: 30 random chunks at the chosen size, each marked "interpretable alone?";
   passes at ≥ 24/30 (80%).

## 6. Tuning (on dev data only)
- Semantic percentile: chosen on Qasper dev so that mean chunk tokens after size enforcement are
  within ±5% of fixed-size 128's mean on the same papers. Segmentation model: MiniLM.

## 7. Predictions
- SciFact: strategies tie at 128; most documents are 1–2 chunks, so there is little room for
  differences.
- Qasper: recursive ≥ fixed (real paragraph structure); semantic ≈ recursive.
- Size: the soft ceiling sits below 254 on Qasper (multi-topic papers) and at or near 254 on
  SciFact (dense single-topic abstracts).
- The title prefix helps on Qasper, where chunks are cut away from the paper they belong to.