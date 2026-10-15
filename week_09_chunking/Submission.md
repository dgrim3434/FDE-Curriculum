# Week 9 — Chunking & Preprocessing

## What I Built

A chunking library with four strategies behind one interface, a metadata extractor, a label-free
chunk analyzer, and a pre-registered retrieval experiment on two corpora with paired and cluster
bootstrap confidence intervals. Everything lives in `week_09_chunking/`; every number below traces
to a file in `results/`.

| Deliverable | Files |
|---|---|
| 1. Chunking library (fixed, sentence, recursive, semantic) | `chunking/base.py`, `fixed.py`, `sentence.py`, `recursive.py`, `semantic.py`, `splitter.py` |
| 2. Metadata extractor (title, authors, date; headings and hierarchy; storage) | `metadata/fields.py`, `structure.py`, `attach.py`, `store.py` |
| 3. Chunk analyzer (size, boundaries, overlap, coherence + baseline) | `analysis/analyzer.py` |
| 4. Comparison experiment | `retrieval/data.py`, `qasper.py`, `index.py`, `metrics.py`, `bootstrap.py`; `experiments/bounds.py`, `sanity.py`, `run_sweep.py`, `compare.py`, `compare_stage2.py`, `floor_audit.py` |

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — dimension 384, `max_seq_length` 256
(254 content tokens after `[CLS]`/`[SEP]`), mean pooling, L2-normalized. Coherence is measured with
a different model, `all-mpnet-base-v2` (768-d), so the semantic chunker is not graded by the model
that placed its cuts.

## Architecture

### One contract for every chunker
Every strategy returns `list[Chunk]` with `doc_id`, `chunk_index`, `start`, `end`, `text`,
`n_tokens`, `metadata`. Eight invariants are tested for all four strategies:
`doc_text[start:end] == text`; starts strictly increase; every non-whitespace character is covered;
overlap only when configured; `n_tokens` re-tokenized from the slice (no special tokens); no empty
or whitespace-padded chunks; deterministic output; empty documents give `[]`. Text is always sliced
from the original, never rebuilt — the span invariant is what makes the metadata join, span-based
relevance and the analyzer possible.

### Design decisions
- **Sizes are in embedder tokens, spans in characters.** The fixed-size chunker windows over
  WordPiece tokens and maps back to characters through the tokenizer's offset mapping.
- **Every structure-aware chunker has a fallback.** A sentence or paragraph longer than the budget
  is split by the fixed-size chunker on its substring, with spans shifted back into the document.
- **Sentence chunker:** greedy packing to a token budget; overlap counted in sentences; overlap
  sentences are dropped oldest-first until the next sentence fits, which also guarantees progress.
- **Recursive chunker:** paragraphs → lines → sentences → token fallback, merging within a
  paragraph; a group below `min_tokens` (the floor) may merge across a paragraph boundary, which
  attaches headings to the text they head. No overlap: cuts land where dependencies are weakest.
- **Semantic chunker:** sentence embeddings, gap distance = 1 − similarity (window of sentence
  vectors), cut above a **per-corpus percentile** of the distance distribution rather than an
  absolute threshold (scale-invariant — tested with an affine transform of the similarities, where
  a hardcoded 0.5 changes the cuts and the percentile does not). Two passes so every sentence is
  embedded once; undersized segments merge toward the smaller cut distance; oversized segments are
  packed by the sentence chunker.
- **Injected dependencies:** tokenizer, sentence splitter and embedder are passed in. Tests use a
  stub tokenizer, a regex stub splitter and a topic-word stub embedder — no model runs in any test.

### Evaluation design (two regimes)
- **Regime A — SciFact (BEIR test, 300 queries):** document-level labels; chunk scores aggregated
  to documents with **MaxP** (a document is relevant if any part is; independent of chunk count);
  nDCG@10 (linear gain, trec_eval/BEIR convention), recall@10, precision@10, RR.
- **Regime B — Qasper test (416 papers, 1,309 evaluable questions):** papers rendered to text with
  one heading line per section; evidence paragraphs located as character spans; search **within
  each question's own paper** (Qasper questions are paper-specific); metric = **evidence recall at
  a token budget** — character coverage of the gold evidence by the chunks that fit in B tokens,
  best annotator per question. A budget, not k, because k chunks of different sizes are different
  amounts of text.
- **Significance:** paired bootstrap on per-query differences (SciFact); cluster bootstrap that
  resamples **papers** for Qasper, because questions about the same paper are correlated.
  10,000 resamples, seed 0. Primary comparisons at Bonferroni level 1 − 0.05/3 = 0.983.
- **Pre-registration:** `results/plan.md` (grid, primary comparisons, decision rules, tie margins,
  predictions) was committed before any retrieval result existed.

## The Four Bounds (results/bounds/bounds.json)

| Bound | Value | Evidence |
|---|---|---|
| **Hard ceiling** | 256 − 2 = **254** content tokens. With an embedded title prefix (p95 = 24 tokens): **230**; title + section (p95 = 35): **219** | read + computed + measured |
| **Soft ceiling** (dilution) | Below 254 on both corpora: size 254 loses to 128 by −0.025 evidence recall on Qasper (CI excludes 0, equal budget fill) and by −0.023 nDCG on SciFact (CI excludes 0) | measured (sweep) |
| **Floor** | p90 sentence length **58** (SciFact) and **52** (Qasper) tokens, so the grid starts at 64 — the smallest size at or above both. On Qasper, 64 already loses to 128 (−0.019, CI excludes 0, equal fill) | measured |
| **Budget** | 4,096-token window, 300 instructions, 50 question, 500 answer reserve, 20-token header per chunk, r = 1 (assumed): 3,246 tokens available → k_max **38 / 27 / 21 / 15 / 11** at sizes 64 / 96 / 128 / 192 / 254. Not binding anywhere in the grid | computed |

Two measurements that motivate chunking at all: with no chunking, **63% of SciFact abstracts** are
truncated and **26% of all SciFact tokens** are invisible to the retriever; on Qasper, **100% of
papers** are truncated and **95% of tokens** are lost.

Running exactly at the ceiling is not safe: fixed-size chunks of 254 still truncated **0.12%**
(SciFact) and **1.7%** (Qasper) of chunks, because a slice starting mid-word re-tokenizes into more
tokens than its window.

## Results

### Sanity checks (results/sanity/sanity.json)
- Reproduction of the whole-document configuration (title + " " + text, one chunk per document,
  no prefix): **nDCG@10 = 0.6451**, matching the published MTEB score for this model on SciFact
  (64.51). The full Week 9 pipeline (loader, embedder, MaxP over single chunks, metrics) reproduces
  an independent number.
- Random ranking: nDCG@10 = **0.000** — the metrics reward only real ranking.

### Regime A — SciFact, nDCG@10 (300 queries)

| Config | nDCG@10 | Recall@10 | Chunks | Coherence |
|---|---|---|---|---|
| whole document (body text) | 0.6402 | 0.7867 | 5,183 | 0.661 |
| fixed 64 | 0.6380 | 0.7972 | 27,955 | 0.692 |
| fixed 96 | 0.6586 | 0.7946 | 19,496 | 0.680 |
| fixed 128 | 0.6623 | 0.8166 | 15,259 | 0.677 |
| fixed 192 | 0.6344 | 0.7986 | 11,126 | 0.670 |
| fixed 254 | 0.6398 | 0.7849 | 8,879 | 0.666 |
| sentence 128 | 0.6644 | 0.8166 | 17,152 | 0.758 |
| semantic 128 | 0.6618 | 0.8132 | 17,577 | 0.761 |

Size comparisons against the best size (128), 95% paired CIs on the difference:

| Size | Δ nDCG@10 | 95% CI | Reading |
|---|---|---|---|
| 64 | −0.0243 | [−0.0502, +0.0010] | inconclusive |
| 96 | −0.0037 | [−0.0250, +0.0172] | inconclusive |
| 192 | −0.0279 | [−0.0484, −0.0081] | 128 better |
| 254 | −0.0225 | [−0.0438, −0.0007] | 128 better |

Strategies at 128 (exploratory): sentence +0.0021 [−0.0153, +0.0197], semantic −0.0005
[−0.0182, +0.0174] — both inconclusive and within ±0.002 of fixed. Chunking at 128 vs the whole
document: +0.0221 [−0.0038, +0.0475], inconclusive at n = 300.

### Regime B — Qasper, evidence recall (1,309 questions, 416 papers)

| Config | ER@256 | ER@512 | ER@1024 | Chunks | Coherence |
|---|---|---|---|---|---|
| fixed 64 | 0.288 | 0.434 | 0.605 | 33,671 | 0.603 |
| fixed 96 | 0.241 | 0.428 | 0.611 | 22,514 | 0.570 |
| fixed 128 | 0.295 | 0.453 | 0.628 | 16,939 | 0.546 |
| fixed 192 | 0.226 | 0.362 | 0.615 | 11,351 | 0.524 |
| fixed 254 | 0.268 | 0.428 | 0.631 | 8,643 | 0.505 |
| sentence 128 | 0.274 | 0.431 | 0.636 | 20,136 | 0.629 |
| **recursive 128** | **0.317** | **0.497** | **0.685** | 24,339 | 0.657 |
| semantic 128 | 0.280 | 0.435 | 0.642 | 20,906 | 0.636 |

Size comparisons against 128 (ER@512, 95% cluster CIs): 64 −0.0185 [−0.0353, −0.0012];
96 −0.0248 [−0.0412, −0.0081]; 192 −0.0909 [−0.1111, −0.0714]; 254 −0.0248 [−0.0470, −0.0029].
Every other size is worse than 128.

**Primary comparisons** (Qasper, size 128, ER@512, cluster bootstrap at 98.3%):

| Comparison | Δ | 98.3% CI | W/T/L | MDE | Reading |
|---|---|---|---|---|---|
| sentence − fixed | −0.0217 | [−0.0458, +0.0035] | 433/368/508 | 0.0224 | inconclusive |
| **recursive − fixed** | **+0.0440** | **[+0.0190, +0.0682]** | 525/390/394 | 0.0231 | **recursive better** |
| semantic − fixed | −0.0177 | [−0.0412, +0.0063] | 443/371/495 | 0.0223 | inconclusive |

Exploratory, other budgets (95%): recursive beats fixed at ER@256 (+0.0217 [+0.0011, +0.0426]) and
ER@1024 (+0.0568 [+0.0387, +0.0751]). Sentence is worse than fixed at ER@256 (−0.0212 [−0.0397,
−0.0021]); at ER@1024 sentence (+0.0078) and semantic (+0.0140) are inconclusive with positive
point estimates.

### Finding: budget fill confounds sizes that don't divide the budget
Evidence-at-budget takes whole chunks only, so a size s can use `floor(B / s) × s` tokens:

| Size | Fill at 256 | Fill at 512 | Fill at 1024 |
|---|---|---|---|
| 64 | 100% | 100% | 100% |
| 96 | **75%** | 94% | 94% |
| 128 | 100% | 100% | 100% |
| 192 | **75%** | **75%** | 94% |
| 254 | 99% | 99% | 99% |

The lowest evidence recall at every budget belongs to exactly the sizes that leave a quarter of the
budget unused (96 and 192 at 256; 192 at 512), and at 1024, where fill is 94–100% for every size,
the spread collapses to 0.605–0.631. The 96 and 192 comparisons at 256 and 512 therefore measure
budget fill as much as chunk size. The conclusion about 128 does not depend on them: **64 and 254
fill the budget as fully as 128 and still lose to it.** The same effect is the likeliest explanation
for sentence and semantic chunking trailing fixed at small budgets but not at 1024 — their chunks
are variable and smaller than the budget, so two of them leave part of a 256-token budget unused.
That is a hypothesis with supporting evidence, not a demonstrated cause.

### Finding: recursive's gain is not just "more, smaller chunks"
Recursive at 128 produced 24,339 chunks against fixed's 16,939 — its chunks are roughly 89 tokens on
average (paragraph boundaries plus the 52-token floor), so more of them fit in a budget. If that were
the explanation, smaller fixed chunks should also do better. They don't: fixed 64 (0.434) and fixed
96 (0.428) both score below fixed 128 (0.453) at ER@512. The gain comes from **where** recursive cuts
(paragraph boundaries), not from how many chunks it makes.

### Metadata extractor
- **Gutenberg front matter vs the Gutenberg catalog (30 books):** title 29/30 correct (coverage
  100%); authors 29/29 correct among books where a name was extracted (coverage 29/30); release date
  30/30 correct. Rules: labeled fields only, `Most recently updated` never used as the date,
  translators never authors, continuation lines attached, partial dates kept as `YYYY-MM`.
  Comparison rules: casefolded title; author surnames as sets after dropping `[Translator]` entries;
  exact ISO date. Per-book rows in `results/metadata/gutenberg_fields.json`.
- **Heading detection on rendered Qasper papers (281 dev papers):** 278 exact; precision 0.999;
  recall 1.000. The false positives are paragraphs that triggered a heading rule; their effect is a
  wrong `section_path` for chunks until the next real heading.

## Decision Defence

### Recommendation: when to use each strategy

| Strategy | Use when | Evidence from this run |
|---|---|---|
| **Recursive** | Documents have real structure (paragraphs, sections) and topics change at those boundaries | +0.044 evidence recall over fixed on Qasper at 128, CI [+0.019, +0.068] at the Bonferroni level; ahead at every budget |
| **Fixed-size** | Short, dense, unstructured text; when speed and predictable size matter | On SciFact, ties sentence and semantic within ±0.002 nDCG@10 |
| **Sentence** | Unstructured prose where readability of each chunk matters | Ties fixed on SciFact; much higher coherence (0.758 vs 0.677); inconclusive vs fixed on Qasper |
| **Semantic** | Long unstructured documents with real topic drift, when its ~2× embedding cost is acceptable | Ties fixed on SciFact; inconclusive on Qasper. No measured benefit here that justifies the cost |

**Ship for Qasper-style documents:** recursive, budget 128, overlap 0, `min_tokens` 52.
**Ship for SciFact-style abstracts:** 128 tokens. Fixed, sentence and semantic are indistinguishable
on retrieval; sentence is preferred on the generator-side evidence (whole sentences, coherence 0.758
vs 0.677) for about 12% more vectors (17,152 vs 15,259).

### The size rule exposed a flaw in itself
The pre-registered rule was "ship the smallest size whose paired CI against the best size includes
zero." On SciFact it selects **64**, because 64's CI is [−0.050, +0.001] — it includes zero by 0.001.
But that CI reads as **inconclusive**, not tied: the data are consistent with 64 being up to five
points worse than 128. The rule conflated absence of evidence with evidence of a tie. Under the
stricter version — the smallest size whose comparison reads as a tie within ±0.01 — no size ties
with 128, and 128 is the choice. I report both: the rule as written selects 64; the size I would
ship is **128**, because 64's apparent tie is a 0.001 overlap on a CI reaching −0.05 and because 64
costs 1.8× the vectors (27,955 vs 15,259) for no demonstrated gain. Next time the rule should be
written in terms of the tie margin, not zero.

### Eight defence questions

**1. What chunk size would you ship, and what are the four bounds that produced it?**
128 tokens on both corpora. The hard ceiling is 254 (and 230 with an embedded title). The floor is
anchored at 64, the smallest grid size at or above the p90 sentence length of both corpora (58, 52).
The budget does not bind (11 to 38 chunks fit in a 4K window). Inside that box the soft ceiling was
measured: 128 beats 254 and 64 on Qasper at equal budget fill, and beats 192 and 254 on SciFact,
with CIs excluding zero.

**2. What overlap, and what semantic unit is it protecting?**
For the shipped recursive configuration: none. Its cuts fall on paragraph boundaries (or sentence
boundaries inside oversized paragraphs), so no sentence is ever split, and paragraph boundaries are
where cross-chunk dependencies are weakest. For fixed-size chunking the unit at risk is a sentence:
without overlap, a p90-length sentence (52 tokens) is cut with probability 52 / 128 = 41% at size
128; an overlap equal to 52 makes that probability zero, at a cost of 52 / (128 − 52) = 68% more
vectors and embedded tokens.

**3. Which strategy, and what is the measured gap to the runner-up?**
Recursive. The runner-up on Qasper at ER@512 is fixed-size (sentence and semantic have lower point
estimates); the gap is +0.044, 98.3% CI [+0.019, +0.068], won on 525 questions, lost on 394, tied on
390.

**4. What would change for legal contracts, API documentation, chat transcripts, scientific papers
with tables?**

| Type | Start with | Bound that moves | Overlap protects | What chunking can't fix |
|---|---|---|---|---|
| Legal contracts | Recursive on the numbered hierarchy (Article → Section → clause) | Floor rises: the unit is a whole clause; splitting it separates an obligation from its conditions | Nothing at clause boundaries | Defined terms and cross-references ("as defined in §1.4"): attach definitions through metadata or expansion |
| API documentation | Recursive on headings; endpoint or function = unit | Soft ceiling rises (one endpoint stays on topic for hundreds of tokens); hard ceiling bites on long code samples | Nothing at heading cuts | Never split a code block; treat it as atomic, falling back only above the ceiling |
| Chat transcripts | Turn- or exchange-based (question → answer) | Floor rises: single turns ("yes, that works") mean nothing alone; sentence splitting is unreliable | One turn | Speaker and timestamp must travel as metadata |
| Papers with tables | Recursive by section (as on Qasper) | Hard ceiling bites on tables: a flattened table is one giant "sentence" | Nothing at section cuts | Chunk tables separately with their captions, or serialize rows; Qasper's table-only evidence was unlocatable in rendered text for exactly this reason |

**5. How would you detect, in production, that chunking is wrong — without labels?**
Index-time: truncation rate of the embedded text (it is not zero even at size 254), below-floor and
dangling-start rates, single-chunk-document rate, and drift in the coherence distribution.
Query-time: the adjacent co-retrieval rate (chunks *i* and *i+1* of one document both in the top-k
means the answers span the cuts), duplicate text in the top-k (overlap too large), and a falling
top-1 score. Coherence is not a usable proxy for choosing a size here: it falls steadily with size
on both corpora (SciFact 0.692 → 0.666; Qasper 0.603 → 0.505) while retrieval peaks at 128.

**6. Retrieval quality is bad. How do you tell whether it is the chunking or the embedding model?**
Two pieces of evidence. The score distribution of relevant vs irrelevant documents diagnoses the
embedder: in Week 8, MiniLM separated relevant SciFact documents from random ones with AUC 0.989, so
the model can recognize relevant text in this domain. The chunker sweep with the model held fixed
diagnoses the chunker: on Qasper, changing only the chunker moves evidence recall by up to 0.09
(size) and 0.044 (strategy) with CIs excluding zero, so chunking is a real lever there; on SciFact
the strategies are within ±0.002 of each other, so the chunker is not the lever on that corpus.

**7. What breaks if you double the chunk size? If you halve it?**
From 128 to 254: retrieval gets worse (Qasper −0.025, SciFact −0.023, both CIs excluding zero) —
the dilution side; truncation appears (0.12% / 1.7% of chunks at 254); the index halves (Qasper
16,939 → 8,643 vectors); each chunk costs twice the generation tokens per slot.
From 128 to 64: retrieval gets worse on Qasper (−0.019, CI excludes zero; inconclusive on SciFact) —
the fragmentation side; the index nearly doubles (16,939 → 33,671 on Qasper; 15,259 → 27,955 on
SciFact) and search time grows with it.

**8. When is chunking the wrong tool entirely?**
When the embedder can't separate relevant text in the domain (no chunking fixes that; change or
fine-tune the model). When questions need the whole document (counting, summarizing a filing) —
map-reduce or long context. When the facts are tabular — a database and text-to-SQL. When answers
span chunk boundaries — retrieve small chunks and expand to neighbors (small-to-big) instead of
re-chunking. And when the document already fits under the ceiling.

### Embed or store metadata?
Store always: title, authors, date, section path and neighbors are needed for citation, filtering
and small-to-big expansion, and cost nothing in the embedding. Embedding a prefix shrinks the
ceiling (to 230 for a title, 219 for title + section at p95) and pulls every chunk of a document
toward the same vector. Its retrieval effect was run (B4) but not analyzed for this submission, so
the shipped default is store-only.

### Coherence metric
Representation fit: the mean cosine between a chunk's embedding and the embeddings of the sentence
pieces inside it — how well one vector represents its parts, which is dilution measured directly.
It misses readability and self-sufficiency. Single-sentence chunks are excluded (NaN), and it is
reported against a random-chunk baseline and against size, because it falls with size.

## Visualizations

The two required figures — the span chart (one Qasper paper under every strategy, sentence ends,
headings, overlap shading and mid-word markers) and the corpus-wide size distribution with the
ceiling, budget and floor lines — are implemented in `viz/phase2.py` but were not rendered or
written up this week. The numbers they would display (size percentiles, boundary alignment,
mid-word cuts, overlap and coherence per configuration) are computed by the analyzer for every
configuration and saved in `results/analysis/`.

### Red Flag Checklist

| Check | Status |
|---|---|
| n stated for every result | Yes: 300 SciFact queries; 1,309 Qasper questions over 416 papers |
| Every strategy given identical settings | Partly. Model, metrics, aggregation, budget and seeds are identical; mean chunk size is not — recursive averages roughly 89 tokens against fixed's 128. Ruled out as the explanation for recursive's gain (smaller fixed chunks do worse), but the comparison is not size-matched |
| Anything measured on data used to tune it | No test labels were used for tuning. The semantic percentile was size-matched on Qasper dev, but for SciFact on the SciFact corpus documents themselves (no labels) |
| A confound in the metric | Yes: budget fill. Sizes that don't divide the budget use 75–94% of it; documented above, and the size conclusion rests on comparisons with equal fill |
| Multiple comparisons | Three primary comparisons with Bonferroni correction; many exploratory comparisons at 95%, reported as exploratory |
| Sample size large enough for the claims | Minimum detectable effect ≈ 0.022 on Qasper; differences smaller than that are reported as inconclusive, not as ties |
| A pre-registered rule changed after seeing results | No. Its flaw (CI-includes-zero conflates inconclusive with tied) is reported, with the rule's own output and the sensitivity alongside |
| Unexplained anomaly | The size-192 dip is explained by budget fill. Truncation at size 254 is explained by re-tokenization of mid-word slices |
| Assumptions stated | Generator window and token ratio r = 1 are assumed, not measured |
| Axes and units labeled | Not applicable this week — figures not produced |

## Testing

Pytest suites in `tests/`, all passing offline with no model loaded: `test_base`, `test_fixed`,
`test_sentence`, `test_recursive`, `test_semantic`, `test_fields`, `test_structure`, `test_qasper`,
`test_attach_store`, `test_analyzer`, `test_coherence`, `test_metrics`, `test_index`,
`test_bootstrap`. Highlights: the eight shared invariants run against every chunker; the window
arithmetic is checked exactly with one-token words; recursive chunking must equal sentence chunking
on unstructured text (an oracle between two of my own implementations); the percentile threshold's
scale invariance is tested against a hardcoded-threshold negative control; the metrics reproduce two
hand-computed nDCG examples and fail if IDCG is built from the ranking; the bootstrap is checked for
calibration (≈95% coverage over 200 simulated experiments) and the cluster bootstrap for being wider
when questions within a paper are correlated. Data loaders and experiment scripts are untested; they
are checked by their outputs (the sanity file, the accounting totals, the size of each split).

## What I Learned

1. **Chunk size is boxed in, not chosen.** Four bounds — a read ceiling, a measured soft ceiling, a
   measured floor and a computed budget — leave a small region worth sweeping. The answer to "why
   128" is a table of evidence, not a convention.
2. **Silent truncation is everywhere.** 63% of SciFact abstracts were truncated by the configuration
   I used in Week 8, and even chunks sized exactly at the ceiling truncate after re-tokenization.
3. **"The CI includes zero" is not "tied".** My own pre-registered size rule picked 64 on a CI of
   [−0.050, +0.001]. A decision rule has to be written in terms of a tie margin.
4. **The evaluation metric can carry its own confound.** Evidence at a token budget penalized sizes
   that don't divide the budget; the pattern across three budgets is what revealed it.
5. **An alternative explanation is only ruled out with a measurement.** "Recursive wins because it
   makes smaller chunks" was answered by the fixed-size sweep, not by argument.
6. **Corpus properties decide what an experiment can show.** On SciFact the strategies could not
   differ (no paragraph structure, short single-topic abstracts); Qasper is where they did.

## Limitations and what is not included

- Stage 2 (block B3, overlap 52 vs 0 for fixed-size at 128; block B4, title prefix vs none for
  recursive at 128) was run and its per-query results are in `results/runs/`, but the comparisons
  were not analyzed for this submission.
- The floor audit (30 chunks each for `B2_recursive_128` and `A1_fixed_64`) was sampled but its
  scores are not reported here; the floor is supported by the p90 sentence measurement and the
  64-vs-128 comparison on Qasper.
- One embedding model only; the model × chunker 2 × 2 was dropped from the plan to keep the week in
  scope.
- The generator-token ratio r is assumed to be 1.
- Visualizations not produced this week.

## Questions

1. Evidence-at-budget with whole chunks creates the budget-fill confound. Is the standard fix to
   allow a truncated final chunk, to evaluate only at budgets divisible by every size, or to report
   recall against tokens actually used?
2. My size rule failed on a CI that overlapped zero by 0.001. In practice, do teams pre-register
   tie margins per metric, or use an equivalence test (two one-sided tests) for "these are tied"?