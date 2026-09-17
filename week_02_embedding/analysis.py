import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Embedding_Analyzer:
    
    def __init__(self, w_in, word_to_id, id_to_word):
        
        self.w_in = w_in
        
        norms = np.linalg.norm(self.w_in, axis=1, keepdims=True)
        
        self.E = self.w_in / norms
        
        self.norms = norms.ravel()
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
    
    def _vec(self, word):
        
        if word not in self.word_to_id:
            raise ValueError("ERROR: Provided word is not in the vocabulary")
        return self.E[self.word_to_id[word]]
    
    def most_similar(self, word, k = 5):
        
        vec = self._vec(word)
        
        # We need to dot product all of the embeddings against the words vector
        
        sims = np.einsum('D, VD -> V', vec, self.E)
        
        # Exclusive Search so we set the words similarity to -inf
        sims[self.word_to_id[word]] = -np.inf
        
        top = np.argpartition(-sims, k)[:k]
        
        top = top[np.argsort(-sims[top])]
        
        return [(self.id_to_word[i], sims[i]) for i in top]
    
    def analogies(self, word_a, word_b, word_c, k = 10, strategy='add'):
        
        if strategy not in set(['add', 'mult']):
            raise ValueError(f"ERROR: strategy must be 'add' or 'mult but was: {strategy}")
        
        v_a = self._vec(word_a)
        v_b = self._vec(word_b)
        v_c = self._vec(word_c)
        
        if strategy == 'add':
            
            v_d = v_b - v_a + v_c
            norm = np.linalg.norm(v_d)
            v_norm = v_d / np.maximum(norm, 1e-9)
            
            sims = np.einsum('D, VD -> V', v_norm, self.E)
        
        if strategy == 'mult':
            
            eps = 1e-3
            s_a = (np.einsum('D, VD -> V', v_a, self.E) + 1) / 2
            s_b = (np.einsum('D, VD -> V', v_b, self.E) + 1) / 2
            s_c = (np.einsum('D, VD -> V', v_c, self.E) + 1) / 2
            
            sims = (s_b * s_c) / (s_a + eps)
        
        # Removing all input vectors from the output
        for w in (word_a, word_b, word_c):
            
            sims[self.word_to_id[w]] = -np.inf
        
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        
        return [(self.id_to_word[i], sims[i]) for i in top]
    
    def generate_random_pairs(self, n = 1000, seed = 0):
        
        rng = np.random.default_rng(seed)
        
        i = rng.integers(0, len(self.E), n)
        j = rng.integers(0, len(self.E), n)
        
        sims = np.einsum('ND, ND -> N', self.E[i], self.E[j])
        
        return np.mean(sims)
    
    def visualizations(self, model_name, top_n=500, seed=42, categories = None):
        
        X = self.E[:top_n]
        labels = self.id_to_word[:top_n]
        
        p2 = PCA(n_components=2, random_state=seed)
        Xp = p2.fit_transform(X)
        print(f"PC1 + PC2 explained variance: {p2.explained_variance_ratio_.sum():.1%}")
        
        X50 = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
        Xt = TSNE(n_components=2, perplexity=25, max_iter=1000,
              metric='cosine', init='pca', random_state=seed).fit_transform(X50)
        
        for name, Y in [('pca', Xp), ('tsne', Xt)]:
            fig, ax = plt.subplots(figsize=(14, 11))
            ax.scatter(Y[:,0], Y[:,1], s=6, c='lightgray')
            if categories:
                colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                for (cat, wl), col in zip(categories.items(), colors):
                    ids = [i for i, w in enumerate(labels) if w in wl]
                    if not ids: continue
                    ax.scatter(Y[ids,0], Y[ids,1], s=70, color=col, label=cat, zorder=3)
                    for i in ids:
                        ax.annotate(labels[i], Y[i], fontsize=9, zorder=4)
            ax.legend(); ax.set_title(f'{name.upper()} — top {top_n} words')
            fig.savefig(f'artifacts/{model_name}_{name}.png', dpi=150, bbox_inches='tight')