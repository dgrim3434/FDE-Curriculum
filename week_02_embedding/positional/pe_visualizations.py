from encoding import sinusiodal_position_encodings
import matplotlib.pyplot as plt
max_len = 250
d_model = 126
import numpy as np
import os

CAT = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
MUTED = "#5c5c5c"
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .5, "legend.frameon": False})

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
def generate_visualizations():
    
    pe = sinusiodal_position_encodings(max_len, d_model)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pe, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("embedding dimension"); ax.set_ylabel("position"); ax.grid(False)
    ax.set_title(f"Sinusoidal positional encoding (d_model={d_model})", loc="left")
    fig.colorbar(im, ax=ax, pad=.015).set_label("PE value", color=MUTED)
    print(parent_directory + "/artifacts/pe_heatmap.png")
    fig.savefig(parent_directory + "/artifacts/pe_heatmap.png")
    plt.close(fig)    
    
    dims = [0, 4, 20, 60]
    fine = np.linspace(0, max_len, max_len * 8)                 
    div  = np.exp(np.arange(0, d_model, 2, dtype=float)         
              * (-np.log(10000.0) / d_model))
    fig, axes = plt.subplots(len(dims), 1, figsize=(10, 1.15*len(dims)), sharex=True, sharey=True)
    for ax, dim, c in zip(axes, dims, CAT):
        w = div[dim // 2]
        y = np.sin(fine * w) if dim % 2 == 0 else np.cos(fine * w)
        ax.plot(fine, y, color=c, lw=2)
        ax.set_ylim(-1.35, 1.35); ax.set_yticks([-1, 0, 1])
        ax.set_ylabel(f"dim {dim}\nλ {2*np.pi/w:,.0f}", color=c, rotation=0,
                  ha="right", va="center", labelpad=12, fontsize=9)
    axes[-1].set_xlabel("position"); axes[-1].set_xlim(0, max_len)
    axes[0].set_title("Each dimension pair is a clock running at its own speed", loc="left")
    fig.savefig(parent_directory + "/artifacts/pe_curves.png")
    plt.close(fig)
    
    E = pe / np.linalg.norm(pe, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(E @ E.T, cmap="Blues")
    ax.set_xlabel("position j"); ax.set_ylabel("position i"); ax.grid(False)
    ax.set_title("cos( PE(i), PE(j) )", loc="left")
    fig.colorbar(im, ax=ax, pad=.02).set_label("cosine similarity", color=MUTED)
    fig.savefig(parent_directory + "/artifacts/pe_similarity.png"); plt.close(fig)
    
    bases, ks = [0, 10, 20, 30], np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for n, (p, c) in enumerate(zip(bases, CAT)):
        y = np.array([pe[p] @ pe[p+k] for k in ks])
        ax.plot(ks, y, color=c, lw=5.0 - 1.1*n, zorder=2+n, label=f"from position {p}")
        ax.plot(ks[n::len(bases)], y[n::len(bases)], "o", ms=5, color=c,
            markeredgecolor="white", markeredgewidth=1, zorder=6+n)
    ax.set_xlabel("offset k"); ax.set_ylabel("PE(p) · PE(p+k)")
    ax.set_title("The dot product depends only on the offset k, never on p", loc="left")
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(parent_directory + "/artifacts/pe_dot_vs_offset.png"); plt.close(fig)
    
    
    print("wrote 4 figures to artifacts/")

if __name__ == "__main__":
    
    generate_visualizations()