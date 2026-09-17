"""
The Skip Gram Class Covers the entire forward pass of the modeling.
"""
import numpy as np
class Skip_Gram:
    
    def __init__(self, V, D, seed = 0):
        
        rng = np.random.default_rng(seed)
        self.V = V
        self.D = D
        
        self.w_in = ((rng.random((V, D)) -0.5) / D).astype('float')
        self.w_in_init = self.w_in.copy()
        self.w_out = np.zeros((V,D)).astype('float')
    
    
    def sigmoid(self, s):
        
        return 1 / (1 + np.exp(-np.clip(s, -30.0, 30.0)))
    
    def forward(self, centers, targets):
        
        # Extracts the embeddings for the center matrices shape: (B, D)
        u = self.w_in[centers]
        
        # Extracts the embeddings for the target matrices shape: (B, K + 1, D): each center has k + 1 targets and each target has an embedding
        v = self.w_out[targets]
        
        # We must now calculate the score for each center target combination which is the dot product of the two embeddings
        
        score = np.einsum('BD, BKD->BK', u, v)
        
        return u, v, score
    
    @staticmethod
    def loss(scores):
        
        pos = np.logaddexp(0, -scores[:, 0])
        neg = np.logaddexp(0, scores[:, 1:]).sum(axis=1)
        
        return (pos + neg).mean()
    
    """
    The step function which covers the complete pass from calculating the score to updating the gradients
    """
    
    def step(self, centers, targets, lr):
        
        u,v, scores = self.forward(centers, targets)
        
        loss = self.loss(scores)
        

        # calculating the labels. For each batch the first column is the positive case
        
        b, k = scores.shape
        
        labels = np.zeros((b, k))
        labels[:, 0] = 1
        
        grad = self.sigmoid(scores) - labels
        
        # The gradient for the u matrix is the grad * v
        # grad matrix = B, K + 1, v matrix shape is B, K + 1, D.
        # The resulting matrix must be shape B, D. One gradient update for each center
        grad_u = np.einsum('BK, BKD -> BD', grad, v)
        
        # The gradient for the v matrix is the grad * u
        # u matrix shape is B, D
        # The resulting matrix must be shape B, K + 1, D. One gradient update for each target embedding for each time it was used
        grad_v = np.einsum('BK, BD -> BKD', grad, u)
        
        np.add.at(self.w_in, centers, (grad_u * -lr).astype("float"))
        np.add.at(self.w_out, targets.ravel(), (grad_v * -lr).reshape(-1, self.D).astype("float"))
        
        return scores, loss
    
    
    # Writing the whole w_in matrix to a text file
    def save(self, prefix):
        
        path = prefix + ".model"
        
        with open(path, 'w', encoding='utf-8') as f:
            
            f.write("Embedding Model V1\n")
            f.write(f"{len(self.w_in)}\n")
            # Looping through each embedding
            for row in self.w_in:
                
                for i in range(len(row)):
                    
                    if i == len(row) - 1:
                        f.write(f"{row[i]}\n")
                    else:
                        f.write(f"{row[i]},")
    
    def load(self, prefix):
        
        path = prefix + ".model"
        
        with open(path, 'r', encoding='utf-8') as f:
            assert f.readline().strip() == "Embedding Model V1"
            
            total_vocab = int(f.readline().strip())
            
            for i in range(total_vocab):
                
                embedding = np.array(f.readline().strip().split(","), dtype='float')
                self.w_in[i] = embedding
        
                