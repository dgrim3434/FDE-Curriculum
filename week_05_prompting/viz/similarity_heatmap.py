"""
Cosine-similarity heatmap for the few-shot selector (Week 5, Visualization #2).

Rows    = test queries (one per intent, confusable intents chosen on purpose)
Columns = the training examples the selector actually picked for those queries,
          de-duplicated and grouped by intent in the same order as the rows
Cell    = cosine similarity(query, example)
Outline = the example's intent matches the query's true intent   ("the diagonal")
Number  = shown only on cells this query's selector actually picked

"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "artifacts" / "similarity_heatmap.png"
OUT_JSON = ROOT / "results" / "similarity_heatmap_stats.json"

# Intents chosen in confusable pairs — the ones your error analysis flagged
QUERY_LABELS = [
    "card_arrival", "card_delivery_estimate",
    "pending_top_up", "top_up_failed",
    "declined_card_payment", "virtual_card_not_working",
    "pending_card_payment", "extra_charge_on_statement",
]
K = 3          # examples per query (keeps the plot readable)
SEED = 42

# Design tokens (light surface, one-hue sequential blue ramp)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def build_matrix(queries_df, train_df, model, E, selector, text_col="text", label_col="label_name"):
    """Returns everything the plot and the stats need."""
    q_texts = queries_df[text_col].tolist()
    q_labels = queries_df[label_col].tolist()

    picks_per_query, q_vecs = [], []
    for text in q_texts:
        rows, _, q = selector(text, train_df, model, E, samples=K)
        picks_per_query.append(list(rows.index))       # original train_df index labels
        q_vecs.append(q)
    Q = np.vstack(q_vecs)                               # (n_queries, d)

    # unique picked examples, grouped by intent in row order, then any other intents
    all_picks = list(dict.fromkeys(i for picks in picks_per_query for i in picks))
    order = {lab: n for n, lab in enumerate(q_labels)}
    all_picks.sort(key=lambda i: (order.get(train_df.loc[i, label_col], len(order)),
                                  train_df.loc[i, label_col]))
    cand_labels = [train_df.loc[i, label_col] for i in all_picks]
    cand_texts = [train_df.loc[i, text_col] for i in all_picks]
    C = E[[train_df.index.get_loc(i) for i in all_picks]]   # (n_candidates, d)

    S = Q @ C.T                                         # (n_queries, n_candidates) cosine sims
    picked = np.zeros_like(S, dtype=bool)
    for r, picks in enumerate(picks_per_query):
        for i in picks:
            picked[r, all_picks.index(i)] = True
    match = np.array([[ql == cl for cl in cand_labels] for ql in q_labels])
    return dict(S=S, picked=picked, match=match, q_labels=q_labels, q_texts=q_texts,
                cand_labels=cand_labels, cand_texts=cand_texts, cand_ids=all_picks,
                picks_per_query=picks_per_query)


def plot_heatmap(m, out_path, title):
    S, picked, match = m["S"], m["picked"], m["match"]
    nq, nc = S.shape
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * nc + 5), max(4.5, 0.55 * nq + 2.5)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    im = ax.imshow(S, cmap=SEQ, vmin=0.0, vmax=1.0, aspect="auto")

    # "diagonal" = cells where example intent == query intent → thin ink outline
    for r in range(nq):
        for c in range(nc):
            if match[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                       edgecolor=INK, linewidth=1.2))
            if picked[r, c]:   # selective labels: only the cells the selector used
                v = S[r, c]
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else INK)

    q_ticks = [f"{lab}  |  {txt[:38]}{'…' if len(txt) > 38 else ''}"
               for lab, txt in zip(m["q_labels"], m["q_texts"])]
    ax.set_yticks(range(nq), q_ticks, fontsize=7, color=INK_2)
    ax.set_xticks(range(nc), m["cand_labels"], rotation=90, fontsize=7, color=INK_2)
    ax.set_xlabel("Training examples picked by the selector (grouped by intent)", color=INK_2, fontsize=9)
    ax.set_ylabel("Test query (true intent | text)", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.5)
    ax.tick_params(colors=MUTED, length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("cosine similarity", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)

    fig.text(0.01, 0.01, "Outlined cell = example has the query's true intent.  "
                         "Number = example the selector actually picked for that query.",
             fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def compute_stats(m):
    S, picked, match = m["S"], m["picked"], m["match"]
    picked_match = picked & match
    top1_hits = []
    for r in range(S.shape[0]):
        cols = np.where(picked[r])[0]
        best = cols[np.argmax(S[r, cols])]
        top1_hits.append(bool(match[r, best]))
    pick_counts = picked.sum(axis=0)                     # how many queries picked each example
    return {
        "n_queries": int(S.shape[0]),
        "n_unique_examples": int(S.shape[1]),
        "k": K,
        "mean_sim_same_intent": float(S[match].mean()) if match.any() else None,
        "mean_sim_other_intent": float(S[~match].mean()) if (~match).any() else None,
        "mean_sim_picked": float(S[picked].mean()),
        "label_precision_of_picks": float(picked_match.sum() / picked.sum()),
        "label_hit_rate": float(np.mean([picked_match[r].any() for r in range(S.shape[0])])),
        "top1_label_match_rate": float(np.mean(top1_hits)),
        "hub_examples": [  # picked for 2+ different queries → possible "generic" examples
            {"label": m["cand_labels"][c], "text": m["cand_texts"][c], "picked_by": int(pick_counts[c])}
            for c in np.where(pick_counts >= 2)[0]
        ],
    }


if __name__ == "__main__":
    from prompting.sampler.sample_generator import top_similarity   # ← the selector to visualize
    from experiments.load_data import load_banking77
    from prompting.sampler.embeddings import embed_pool
    train_df, _ = load_banking77(testing_size=50, seed=SEED)  
    test_df = pd.read_csv(ROOT / "data" / "banking77_test.csv", encoding="utf-8")
    
    E, model = embed_pool(train_df, "text")

    # one test query per intent, in QUERY_LABELS order (seeded → same rows every run)
    queries = pd.concat([test_df[test_df["label_name"] == lab].sample(1, random_state=SEED)
                         for lab in QUERY_LABELS], ignore_index=True)

    m = build_matrix(queries, train_df, model, E, top_similarity)
    plot_heatmap(m, OUT_PNG, f"Query vs selected-example similarity (top_similarity, k={K})")
    stats = compute_stats(m)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"saved {OUT_PNG}")