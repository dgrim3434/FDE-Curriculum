# Week 2 - WordtoVec Embedding Model From Scratch

**Corpora:** text8 (primary, Kept first 5,000,000 tokens) , prose corpus (smaller corpus for testing)

---

## WHAT I BUILT

A Skip-gram embedding model with negative sampling from scratch using NumPy. The Architecture begins by ingesting a vocabulary corpus which must already exist within the repository. The vocabulary is then broken into words in order to create the actual vocabulary where the top K most frequent words are selected and assigned an index from 0 to Vocab_len - 1. The decision to use whole words instead of the tokens generated from my week 1 tokenizer was a design decision which was choosen in order to increase the interpretability of the top_similar and the analogies methods which would return chunks of words depending on how the tokenizer split them. After the vocabulary is created we do subsampling of the vocabulary in order to remove some of the most common words within the corpus in order to create more meaningful pairs later on. The next peice is moving into the sampler class which has two important methods first being the pair generation which is responsible for generating the contex of each center word and the second being the negative sampler which samples words randomly from the dataset based on how frequent they occur in the corpus raised to the ^.75 power which decreases the probability gap between rare and more common words. Once this is done we have the pairs and negative samples which is all we need in order to begin training. During training the models goal is to differentiate between the actual pair and the randomly selected negative words. Once the model has been trained the w_in matrix is extracted and holds the vectors for each of the words within the corpus. This is the key idea of the wordtovec architecture we do not actually care about the predictions but instead care about the learned weights. Although dot product was used to caclulate the gradient when selecting the most similar words we used the cos similarity so the w_in matrix was normalized.PCA and t-SNE where also run on the final embeddings to get a better understanding of which directions held the most meaning and which words where the most similar within the space. When interpreting the results PCA retains the overall distance of the words but simply shrinks the dimensions they are represented in but t-SNE does not maintain distance thus you should not read deeply into the distance between words but instead observe the clusters as a whole. It's also important to understand that similarities on there own do not carry much meaning. It's the same idea as reporting a machine learning model without comparing it against benchmark results. To solve this issue a random-pair function was created which randomly samples words within the vocabulary and calculates there similarity and reports the mean across all of the samples. This allows you to compare a score against the random mean which allows you to truely see how high of a similarity score you have. It also aids in detecting antisotrapy which can happen within vector spaces where all words are extremely similar. Lastly sinousodal positional encodings where implemented which allows the model to understand the tokens position within a sequence. This is one of the key tradeoffs when using transformers instead of RNNs they trade the natural positioning which occurs when the model works in sequence for the increased performance with working in parallel which means positioning must be injected into the model. For the main experiment the model was trained using text8 data using a vocab size of 15,000, dimensions of 100, context window of 5, and 5 negative samples the final loss at the end of training was: 2.28

---

## 2. ARCHITECTURE

### 2.1 Decisions and why

    - Skip Gram was choosen over CBOW for two main reasons. The first was because of the corpus size constraints and the fact that Skip Gram usually performs better on smaller corpuses given that each center word recieves 2 x context_window more training examples compared to CBOW. Secondly I wanted to capture deeper semantic meanings within works for example capturing royalty which allowed the analogy king : man :: woman : ? to return queen which is what made wordtovec popular. In order to do this I needed to ensure even rare examples recieved enough training for the semantics to be captured. For people wondering why CBOW would be choosen instead would be when working with a large corpus and you want more stable gradients since CBOW takes the mean of the context and uses that to predict the center word.

    - Within training two matrices where used where w_in represented the embedding when the word was the center and w_out was the embedding when the word was in the context. The main reason for this is it allows for much greater flexability in how the embeddings can be represented especially for common words which will appear often within the negative samples when using one matrix the model needs to minimize a.dot(a) and since the vector a is the same the only way to decrease the loss is the decrease the magnitude of a which results in worse overall embeddings.

    - Negative sampling was used over full softmax for effieciency reasons. Using full softmax will likely produce the best overall predictions but comes at a significant compute cost because in the denominator you must calculate the probability for each word in the entire corpus. As you can see this is extremely costly and as the corpus increase it becomes completely infeasable. The full softmax would answer the question of given this center word what is the context but we instead change the question to is this center, context pair likely given what we has seen within the data. But if we where to only train the model on actual pairs the model would learn that to minimize the loss all it has to do is set the vectors to the same infinitely large value and the loss would go to 0 but no real information would be learned. This is where Negative sampling comes in where we randomly select words and mark them as negatives so instead of sampling the entire vocabulary for each center word if you are using 5 negative samples you only look at 6 total words for each center which is much more efficient and the results are still good.

    - Using cosine similarity instead of dot product when selecting similar words. The model is trained using the raw dot product and it's usually a best practice to use the same metrics which where used during training when caclulating similarity because the model stores information in the magnitude of the vectors. Cosine similarity was choosen because if you use the raw dot product it will inflate the similarity between words and the most common words in the corpus which isn't what we are looking for when looking for the most similar words. This is because a large vector magnitude represents that the model saw the vector very often and that it's very confident in the prediction which is important but will lead to worse overall results.

    - The Dynamic sliding window for pair selection was used because it allows for weighted averages to be computed without additional computational overhead. This might take a second to sink in but since we compute the context for each center word for each epoch it allows words which are closer in length to the center word to have a higher probability of being selected compared to words which are further. Words which are selected more often will be trained more and will have a larger impact on the words final embedding. The allows for a weighted average based on distance from the center word to be leveraged without an additional overhead and ensures closer words will have a greater affect on the center.

### 2.2 The loss and it's gradient

Within training the similarity of two words is measured by the dot product of it's vectors with higher dot products signaling the words are extremely common and lower magnitudes meaning the words are not similar. The dot product was then passed into the sigmoid function which convered the dot product into a probability value between 0 and 1. The sigmoid result is then covered into a log value which represents the loss. So for positive contexts the loss = - log(sigmoid(+score)). We know that -log(1) equals 0 and -log(0) equals INF which makes sense for positive terms. For the negative samples the opposite is true we need to punish high scores and reward low scores. So the loss for negative samples is log(sigmoid(-(-score))). If the dot product is extremely high it gets turned into a extremely large negative number and the sigmoid result is a number near 0 and log(0) is INF. So the final loss equation is -log(sigmoid(+score)) - summation of all k negative samples log(sigmoid(-(-score))). When you take the derivative the result is sigmoid(score) - target. Where the target is 1 if the word is from the context and 0 if the word is from the negative samples.

### 2.3 Data Flow

1. You start with a batch of center words shape: (B,)
2. For each center word you have K + 1 context words(k negative samples 1 actual pair) shape: (B, K + 1)
3. You extract the embedding assocaited to each of the center words within the batch. Each embedding has D dimensions shape: (B,D)
4. You extract the embedding assocaited with each word within the context. Shape: (B, K + 1, D)
5. You need to calculate the score of all pairs which is the dot product between the center and context embedding. Shape (B, K + 1)
6. Take the sigmoid of each of those scores and subtract the target which is either a 1 or 0. This gives the gradient which is (B, K + 1)
7. Next you need to calculate the gradient to update the center matrix. score = u.dot(v) (where u is the center matrix)
    - The derivative in terms of u is v. so the u update is gradient * v.
8. Lastly you need to calculate the gradient to update the context embeddings which is the opposite and is gradient * u.


## 3. CODE

**REPO Link:** https://github.com/dgrim3434/FDE-Curriculum/tree/main/week_02_embedding

**step function:** Responsible for calculating the scores, loss, gradients and updating both the centers embedding and the context embedding. 
```python
def step(self, centers, targets, lr):
        
        u,v, scores = self.forward(centers, targets)
        
        loss = self.loss(scores)
        
        b, k = scores.shape
        
        labels = np.zeros((b, k))
        labels[:, 0] = 1
        
        grad = self.sigmoid(scores) - labels
        grad_u = np.einsum('BK, BKD -> BD', grad, v)
        grad_v = np.einsum('BK, BD -> BKD', grad, u)
        
        np.add.at(self.w_in, centers, (grad_u * -lr).astype("float"))
        np.add.at(self.w_out, targets.ravel(), (grad_v * -lr).reshape(-1, self.D).astype("float"))
        
        return scores, loss
```

**loss function:** The function which calculates the loss at each step. np.logaddexp takes the np.exp of each parameter adds them together and takes the log. This is the same calculation as the log sigmoid. This function is more stable and handles cases where the result is 0.
```python

    def loss(scores):
        
        pos = np.logaddexp(0, -scores[:, 0])
        neg = np.logaddexp(0, scores[:, 1:]).sum(axis=1)
        
        return (pos + neg).mean()
```

**Sinusoidal_positional_encoding():** Creates the positional vectors which will be added to the semantic embeddings trained by the model before entering the attention layer.

```python

    def sinusiodal_position_encodings(max_len, dims):
    
        assert dims % 2 == 0
        pe = np.zeros((max_len, dims), dtype=np.float32)
        pos = np.arange(max_len, dtype=np.float32)[:, None] 
        twoi = np.arange(0, dims, 2)
        dems = np.exp(twoi * (-np.log(10000) / dims))
    
    
        pe[:, 0::2] = np.sin(pos * dems)
        pe[:, 1::2] = np.cos(pos * dems)
    
        return pe
```

## 4. DEMO

### 4.1 Training log

Here is the training log for the final epoch of the text8 model:
```
ep4 step540000/549015 lr=0.00051 loss=2.2971 pos=-1.023 neg=-2.059 |W_in|=2.266
ep4 step541000/549015 lr=0.00046 loss=2.2959 pos=-1.021 neg=-2.061 |W_in|=2.266
ep4 step542000/549015 lr=0.00042 loss=2.2956 pos=-1.020 neg=-2.060 |W_in|=2.267
ep4 step543000/549015 lr=0.00037 loss=2.2968 pos=-1.022 neg=-2.058 |W_in|=2.267
ep4 step544000/549015 lr=0.00033 loss=2.2967 pos=-1.024 neg=-2.059 |W_in|=2.267
ep4 step545000/549015 lr=0.00028 loss=2.2999 pos=-1.024 neg=-2.058 |W_in|=2.267
ep4 step546000/549015 lr=0.00024 loss=2.3049 pos=-1.035 neg=-2.057 |W_in|=2.267
ep4 step547000/549015 lr=0.00019 loss=2.2946 pos=-1.019 neg=-2.060 |W_in|=2.267
ep4 step548000/549015 lr=0.00015 loss=2.2965 pos=-1.023 neg=-2.059 |W_in|=2.267
ep4 step549000/549015 lr=0.00010 loss=2.2979 pos=-1.021 neg=-2.058 |W_in|=2.267
```

Here is the training log for the final epoch of the prose model:

```
ep4 step16000/19710 lr=0.00479 loss=0.6376 pos=-0.366 neg=-0.410 |W_in|=1.190
ep4 step17000/19710 lr=0.00352 loss=2.6354 pos=-1.516 neg=-1.696 |W_in|=1.192
ep4 step18000/19710 lr=0.00226 loss=2.6368 pos=-1.519 neg=-1.697 |W_in|=1.193
ep4 step19000/19710 lr=0.00100 loss=2.6366 pos=-1.518 neg=-1.694 |W_in|=1.193
```

From these results you can see that the text8 model does achieve a lower loss but not by the margin which you would expect given how much more data the text8 model was trained on. This is a great example for why loss on it's own can lead you to the wrong conclusion when looking at the different between the positive and negative samples loss you can see there is a much more defined gap between the two within the text8 model compared to the prose model. This is the most important peice because from these logs we can see that the prose model is simply labelling all words as similar where as the text8 model is actually able to differentiate between noise and similar words. This difference will become evident when looking at the nearest neighbours and analogies produced by these models. For the prose model you can see that the loss goes from 0.6376 to 2.6354 between steps 160,000 and steps 170,000 this is a significant change and highlights the instability within the corpus space with both pos and neg loss being almost the same and extremely low.


### 4.2 Nearest neighbours

The same words where selected and the top 3 most similar words within the embedding space where selected. The first model was trained on text8 data and the second model was trained on prose data. Both models recieved the same exact hyperparameters but the prose corpus only allowed for 4,600 words to be passed into the vocabulary where as the text8 model made full use of all 15000 vocabulary positions. The text8 model was trained on a total 549000 steps where each step trained the model on 128 word centers compared to the prose model which was only trained on 200,000 steps which is a significant drop off which we would expect to show in the results.

| Model | input word | Word 1 | Word 1 Similarity Score | Word 2 | Word 2 Similarity Score | Word 3 | Word 3 Similarity Score |
| --- | ---: | ---: | ---:| ---: | ---: | ---: | ---:|
| Prose | king | august | 0.999 | pieces | 0.999 | beale | 0.999 |
| Text8 | king | consort | 0.788 | throne | 0.775 | isabella | 0.753 |
| Prose | water | hung | 0.999 | seemingly | 0.999 | forming | 0.999 |
| Text8 | water | pouring | 0.733 | brackish | 0.732 | inflow | 0.726 |
| Prose | one | mortal | 0.999 | equal | 0.999 | picture | 0.999 |
| Text8 | one | six | 0.959 | eight | 0.953 | five | 0.942 |
| Prose | france | possess | 0.999 | hollow | 0.999 | laborers | 0.999 |
| Text8 | france | italy | 0.778 | spain | 0.763 | nantes | 0.744 |

You can see that the results from the two models are extremely different. In order to interpret the similarity scores from the model you must first understand the average similarity between two random words within the corpus. For the prose model the average similarity was 0.998 and the average similarity of the Text8 model was 0.367. These are two completely different numbers and the prose model has simply learned the best way to decrease the loss is by setting all words to be similar to eachother. This is a common issue in embedding spaces and is called anisotropy. This happens because there isn't enough signal shown between actual pairs and noise which is commonly the case with small corpuses like the one the prose model was trained on. This has a significant affect of the embedding space as you can see the top 3 similar words are mostly nonsense. On the other hand the text8 model has an averge similarity of .367 which is very low in comparison and means the model was able to seperate the noise of randomly selected words from ones that actually occur near each other. When looking at the results you can see this is the case with all of the similar words making sense and means the similarty score carries meaning. When looking at the table you see the text8 model similarity score is mostly between .8 and .75 which means the model is confident in the similarity between the two words. There is one outlier row which corresponds to the word one which is a number and all of the similarity scores are much higher between 0.96 - 0.94 this means the model was extremely well trained on numbers and is extremely confident in there similarity.

### 4.3 Anisotropy contrast

It's important to understand why the anisotropy which was observed within the prose model is so harmful. When running words through the embedding model our goal is to assign a semantic meaning to each word. The WordToVec uses the distribution hypothesis which states that similar words will have similar contexts(similar words around them) this is the whole idea of the embedding model. In order to have similar words you must also have words which are not similar so when you have extreme anisotropy like observed within the prose model it renders the embedding space completely unusable. Another important peice of anisotropy is the inflated similarity scores here we saw an extreme case which was extremely easy to detect given the average similarity of two random words is 0.998 but anisotropy can show up in more subtle fors where the average similarity is 0.7 which doesn't immediately flag the embedding space as unusable but instead provides a baseline which must be used when selecting whether two words are actually similar. This can be done using the Z score which equals (score - mean_random) / s.t.d_random which allows you to see how much the similarity deviates from the random distribution.

### 4.4 Analogies

Analogies take the input of three vectors and compute b : a :: c: ?. In english this is comuting b is to a what c is to ?. The analogies is one of the things that caused the rise in popularity of WordToVec embedding model because when trained correctly it can provide results like 'king is to man what woman is to queen'. There are two different calculations which are done for each model the first is 3CosAdd which is: b_vec - a_vec + c_vec. The second is 3CosMul which takes each vector and multiplies it by the embedding matrix and then does (b_res * c_res) / (a_res + eps). The difference between the two is that pure vector addition can cause one vector to control the projection space where as the multiplication attempts to balance the weight of each vector. It should be expected that 3CosMul will be more stable when the model was trained on a smaller corpus but since we have already seen the anisotropy within the prose model it's likely that neither of the methods will perform well.

| Model | Analogy Method | word_a, word_b, word_c | Word 1 | Word 1 Score | Word 2 | Word 2 Score |
| --- | ---:| ---: | ---: | ---: | ---: | ---: |
| Prose | Add | one, two, three | flew | 0.998 | each | 0.997 |
| Prose | Mult | one, two, three | flew | 0.998 | each | 0.998 |
| Text8 | Add | one, two, three | four | 0.897 | five | 0.864 |
| Text8 | Mult | one, two, three | four | 0.988 | five | 0.979 | 
| Text8 | Add | run, running, walk | walking | 0.705 | tore | 0.660 | 
| Text8 | Mult | man, king, woman | queen | 0.919 | consort | 0.913 |


As you can see for the prose model anisotropy is overpowering the results since all vectors are extremely similar the embeddings are not able to seperate noise from true similarity which leads to the poor results. The Text8 model performs very well and properly selected the most likely word for all 3 test cases. For the difference between 3CosAdd and 3CosMult you can see the mult method produces slightly higher similarity scores compared to the add method both where correct in the most similar words with the only difference being the assigned similarity score.

## 5. WRITTEN EXPLANATION - NEGATIVE SAMPLING

In order to use the softmax function within training it requires V-way classification. In order to do V-way classification you must compute the probability for every single word within the vocabulary. This is extremely expensive and not feasable when you are working with a large vocabulary. Instead of V-way classification the problem was convered into binary classification so given a (center, context) the model predicts where the pair actually happens are is noise. This switch is simply to optimize the denomination which in V-way classification is a summation of all the probabilities within the vocabulary and by converting into binary classification each prediction no longer depends on the rest of the vocabulary and because much more efficient. This raises an issue because if the model is soley focused on predicting true or false for a given (center, context) pair we must include pairs which are not true. This is where negative sampling comes in and we select words at random based on how common they occur within the dataset. The amount of negative samples selected for a given pair of (center, context) is denoted as k. So for a given pair we compute the probability of pair occuring for k + 1 pairs. For the positive word the loss is = - log(sigmoid(score)) for a negative pair the loss is = log(sigmoid(-score)). This difference is because for positive cases we want very low loss when the model is correct so when the sigmoid is close to 1. For negative pairs we want the similarity to be low which is why the negative sign is moved inside the sigmoid so high values are punished harder since log(0) goes to INF. When sampling for specific words we use the words frequency raised to ^.75. What this does is slightly decreases the probability for extremely common words and increases the probability of less common words. This is important because we want a balance between very common and rare words within the negative samples. The result from using negative sampling with k samples is the w_in matrix represents the PMI - log(k). The PMI measures how similar two words are compared to random chance. As you can see this is exactly what we are asking the model to do which is differentiate between randomly selected words and the pair which we know is actually true.

## 6. What I Learned

    - The importance of selecting the correct learning rate, batch_size combination. For the first iteration I trained the model using a learning rate of 0.05 and a batch_size of 250 but the gradient exploded and I was pretty confused why since the original documentation uses 0.05. What I realized is that since the gradients only get updated after each batch the two hyperparameters must be in sync. When looking back at the documentation I found that they used a batch size of 128 with the learnign rate of 0.05 when I made this change the exploding gradient disappeared. 

    - The importance of testing even the simplest lines of code. For the first iteration of modeling I had written the sigmoid equation incorrectly and hadn't written any test cases in order to ensure the sigmoid function was working as expected. The sigmoid function is extremely simple to the point where I didn't think I needed to test it which ended up costing me time later on when I was searching for errors in my codebase which would have been caught by one test case which would have shown the sigmoid function wasn't producing the expected output. From now on I will be writing test cases based on how often a function is used instead of how complex it is. For the sigmoid function it's called after each step which means it's extremely important and must be tested thoroughly no matter how simple it is.

    - The loss function alone doesn't provide enough information to understand how the model is actually performing. This is especially true within an embedding model and was seen first hand within the results. The prose and text8 models didn't have significantly different final loss values but the performance was significantly different. The prose model was able to achieve a final loss of 2.6 which seems fine until you see that the model was pulling all embeddings toward each other which could be seen instantly when comparing the loss between positive and negative samples.

## 7. Questions / Uncertainties

    - For the attention layer which is next week I am still slightly confused how the sinousodal embeddings allow it to fully capture position. I understand the idea behind it and that it allows the model to see offsets within the position so that it's able to understand where it needs to go but overall I am still a little confused about the entire concept

    - For domain specific tokenizers and embedding models how important is the balance between domain specific text and regular text. Especially for a chat bot if you only train on domain specific text I would assume it would struggle handling user input which doesn't match the domain.

    - For RAG models using domain specific text I wonder what kind of setups you could come up with because the domain specific embeddings are extremely important and the query being structured in a way that replicates the text the model was trained on is also important. I can imagine seperating the chat bot model from the embedding rag model could provide significant increases in results.

    - Since I am specifically interested in domain specific models I have a lot of open questions about the process which must be done in collecting and training models. I have seen a lot of information about learning from teacher models and wonder how that translates the tokenizer and embedding models. Given the teacher model has a significantly larger vocabuluary what kinds of techniques are used to select the best vocabulary for the domain while also keeping the total size small. I can image there are a lot of cool techniques for this that can provide great results at a fraction of the price.