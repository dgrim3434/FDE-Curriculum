from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # save to file, no pop-up window
import matplotlib.pyplot as plt
import pandas as pd


def plot_confusion_matrix(df: pd.DataFrame, out_path: Path, title: str) -> pd.DataFrame:
    """df needs columns true_label, pred_label (None = invalid output)."""
    pred = df["pred_label"].fillna("INVALID")
    labels = sorted(set(df["true_label"]) | set(pred))
    cm = (pd.crosstab(df["true_label"], pred)
            .reindex(index=labels, columns=labels, fill_value=0))

    size = max(6, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=7,
                        color="white" if v > cm.values.max() / 2 else "black")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def top_confusions(df: pd.DataFrame, n: int = 10) -> pd.Series:
    """The most frequent (true → predicted) mistakes, as a ranked list."""
    pred = df["pred_label"].fillna("INVALID")
    wrong = df.assign(pred_label=pred)[df["true_label"] != pred]
    return (wrong.groupby(["true_label", "pred_label"]).size()
                 .sort_values(ascending=False).head(n))