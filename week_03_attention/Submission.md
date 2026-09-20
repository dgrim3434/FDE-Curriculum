# Week 3 - Attention Mechanism
This week I built all of the mechanisms behind attention within the transformer architecture. Over the previous two weeks the tokenizer and embedding model worked to create and provide meaning to individual tokens. The attention layer is the peice which provides context to the token.

## What I Built

There are four main components that make up the attention layer. The first peice is the softmax function which converts a row of scores into individual weights which sum to one. What makes the softmax function I built different from pytorches is mine guards against two potential errors float overflow and propogating NaN values which are caused by left padding which can destroy a training run. The second peice is the causal padding which is extremely important to prevent leakage during training and is the main mechanism which allows for prompt caching. Causal padding ensures tokens can only see ones which have happened before them. During training token 0 is tasked with predicted the next token if it's able to see the next token this prediction is easy which will cause the loss to decrease even though nothing was truely learned. The third piece is the scaled dot product attention which holds all of the actual attention computation. Within this function we use three seperate trainable matrices Q, K, V. Each of these matricies serve a unique purpose the Q matrix is the query matrix and it is trained to ask the question what does this token need. The K matrix is trained to provide the answer of what this token has. We then take the dot product of the rows the Q matrix agains the rows of the K matrix which provides use with scores telling us if the token i can provide token j with the information which it needs. These scores are divided by the square root of d_k to ensure the standard deviation of the scores remain around one which ensures the softmax performs well. After the softmax these weights are then dot producted one more time with the V matrix which holds the actual information which each token has to offer. These results are then passed into the residual stream and added to the tokens embeddings. The last peice is the single and multi-head attention. Each head of attention can be thought of having the ability to answer one specific question so the idea is that by increasing the amount of heads you are able to have each of them specialize for specific information they are looking for. Research proves that heads do specialize in gathering specific information but some of them are redundant and can be pruned at minimal cost to overall performance. Since there was no training within this week I relied on the 41 test cases which where written. Some of these tests compared against pytorches implementations of attention to ensure my calculations matched. Others checked specific edge cases which could sneak past the pytorch tests since errors could sneak past.

## Architecture Decisions

    - One of the main architecture decisions made this week was to keep all math as seperate as possible. This ensured that each peice could be thoroughly tested before moving onto the next. After learning the hard way last week with my softmax function causing hours of searching for bugs I wanted to ensure I could catch implementation errors as quick as possible. This meant narrowing the search space as much as possible which meant fully testing the scaled dot product attention math before moving on the SHA and MHA.

    - The softmax function was build to guard against rows which are fully masked. This occurs when using left padding and causal masks and can result in NaN calculations within the softmax function. This is extremely harmful since any calculation done against NaN results in NaN and it poisons the weights. Guarding against this means that left padding can he used without worrying about degraded performance.

    - Returning the output weights for each layer was a decision done to ensure the weights could be inspected which is extremely important for debugging and understanding the behavior within a specific layer. When it comes time for training and inference this will be an optional return and will default to false to ensure everything is as memory efficient as possible.

    - Currently the attention layer requires both the x and context share the same dimensions. This was done for simplisity sake since my main focus was understanding the math behind attention. This is not a requirement within a transformer and would require additional code changes for context inputs with different dimensions. LLMs which are my main focus use self-attention so this would not be an issue but for any architecture using cross-attention this could cause issues.

## Code

**Repo:** https://github.com/dgrim3434/FDE-Curriculum/tree/main/week_03_attention

**Causal Mask:**

```python
def causal_mask(d_q, d_k = None, device = None):
    
    # If d_k is none it's self-attention
    d_k = d_q if d_k is None else d_k
    
    return torch.tril(torch.ones(d_q, d_k, dtype=torch.bool, device=device), diagonal= d_k - d_q)
```

**Scaled Dot Product Attention:**
```python
def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None, single_head = False):
    
    d_q = Q.shape[-1]

    if single_head:
        scores = torch.einsum('BQD, BKD -> BQK', Q, K) / math.sqrt(d_q)
    else:
        scores = torch.einsum('BHQD, BHKD -> BHQK', Q, K) / math.sqrt(d_q)
    
    if mask is not None:
        
        scores = scores.masked_fill(~mask, float('-inf'))
    
    weights = stable_softmax(scores,dims =-1)
    
    if dropout is not None:
        
        weights = dropout(weights)
    
    return weights @ V, weights
```

**Single Head Attention:**
```python
class Single_Head_Attention(nn.Module):
    
    def __init__(self, d_model, d_k = None, bias = True):
        super().__init__()
        d_k = d_model if d_k is None else d_k
        
        # Defining Linear transformations
        self.W_q = nn.Linear(d_model, d_k, bias=bias)
        self.W_k = nn.Linear(d_model, d_k, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
    
    def forward(self, x, context = None, mask = None):
        
        context = x if context is None else context
        
        return (scaled_dot_product_attention(self.W_q(x), self.W_k(context), self.W_v(context), mask=mask, single_head=True))
```

## Visualizations

### Single-head attention heatmap

The single-head attention heatmap shows the query tokens on the y-axis and the key tokens on the x-axis. The way this visualization should be interpreted is that a single point allows you to see how much weight a specific tokens key played into it's another tokens query update. Some invariants of this heatmap is that summing across the rows should sum to 1. For the first token it should always have a weight of 1 on itself unless the first token is padding. This visualization allows you to visualize the causal padding to ensure the function is opperating correctly. Although your test functions should have caught any errors before it because time for visualization it will allow you to see if the causal padding is working correctly. It's important to understand the weights only represent where information could flow and not what actually drove the output. This is because the weights are multiplied by the value vectors so a high weight but a low vector norm will still result in a small update which is not visible within the heatmap. The second peice is that information gets passed through all of the tokens thus you can not say with certainty which token had the most weight in the output. It's also important to understand given the architecture of the transformer it creates sink tokens which is usually the first token within the sequence and will recieve a high weight in the output which could lead you to believe the model soley focuses on the first token. This happens because of the softmax function where it must allocate weights summing up to 1 even if no relevant information is present and since the first token is visible to all tokens after it the weight is usually inflated and this is not a bug. It's also important to understand weights are only retreivable within open weight models so when using an API this visualization cannot be used.

### Multi-head grid

This grid shows heatmaps of the same structure as the single head attention heatmap and allows you to visualize differences between heads. This can be especially helpful when attempting to visualize specialized heads which have been discussed a lot throughout the research space. When working with large models it's impossible to visualize each head at the same time which is why calculating metrics like entropy, and last-token weight are important to understand the behavior of the head before deciding which ones should be visualized. The Multi-head grid I created was on untrained matrices and since each of these matrices was randomly initialzed with similar values it's to be expected that each head would perform similarly. The key phrase is similar if the heads are fully bit identical then you have a big within your pipeline. This is another example of a bug which should not be diagnosed within a heatmap because it's unlikely you will be able to see bit identical heads which is why you need the proper assertions within the testing suit to catch this help of an issue. This test would be caught when comparing against the pytorch function but since we will not always have prebuilt functions which we can use it's important to really understand why this problem could happen. An issue like this would happen if you did something like expand and then broadcasted which would cause each head to point to the same tokens. This is an issue which should also be caught by your leakage test which checks the diagonal to ensure no columns above it are populated. Post training if the heads are still extremely similar this is a sign that some pruning can be done. When the wieghts are similar across heads it means the heads might be redundant. As we have said before heads can be specialized which increases the importance for testing the heads on diverse sets of prompts. One head might appear to be completely redundant when tested on one set of prompts and be extremely important on another. When pruned the performance issues will not arise until you test on the kind of prompts which the head has been specialized for. Before doing any pruning you should consider the cost/latency benifits against the performance decrease. You should not prune the model just to prune it but instead be looking to save cost and latency. Lastly when observing the weights of each key vector it's important to compare them to the uniform baseline which can be calculated as 1/ (i + 1) Where i represents the position of the token within the sequence. If the weights are all centered around the uniform average the head isn't distingiushing important tokens but instead assigning a uniform weight across them.

## Why multi-head helps

Multi-head attentions assigns a head to specific dimensions within the modeling space. So one head might attend to the dimensions 0-12 and the next one 13-24. When working with a single head there is a single query and a singular softmax function which has the constraint that the weights must sum to one. When a specific token has a lot of information which it's looking for a single query + softmax isn't able to provide the token with the information it's looking for but instead geared to search for one specific peice of information. Lets say the token has two peices of information which it's looking for which would cause two other tokens to recieve high scores. These two scores would then be passed into the softmax which would average them out to weights between 0.4 and 0.5. We then take the weighted sum which causes both of these peices of information to be blurred together. Multiple heads allow specific heads to focus on specific peices of information. Lets say we had four heads which would allow four unique peices of information to be retrieved from other keys and would each be passed into there own softmax which would allow them to recieve much higher weight than if they where combined into one head. Since heads are assigned to specific dimensions within the vector space the overall parameter count is the exact same as if we used one wide head. The cost comes in the rank of these vectors. If you have two many heads each head will be assigned a low dimensional space and be unable to extract true meaning since it's low dimensional. This is an important tradeoff and why more heads doesn't automatically mean better results. After each head passes through the matrices the weight are multiplied by the value matrix and the results are concatinated back together so that each row once again corresponds to a full token. This is mostly bookkeeping and allows for later results to be computed easier. Since each head corresponds to unique dimensions within the vector space you can think of it as cutting each token into k equal peices and then glueing the tokens back together at the end of the layer. Many of the heads within a transformer are likely to be redundant and not add any value. The heads show the most benifit during training since they allow for unique queries to be explored at the same time. During inference there can be many redundant heads which can be pruned without affecting the performance significantly. Pruning is something that must be done with care since there are specialized heads which might look completely redundant within specific domains but are essential within others. Because of this you must clearly define how the model is actaully being used and which kinds of prompts it will actually see in production. Until this is done you are blindly pruning the model without understanding how it will affect performance later on. Pruning is a level which can be pulled to decrease both latency and cost so defining the use cases will allow you to clearly measure the performance impact and make much better decisions on which heads are actually needed.

## What I Learned

    - Throughout each of the weeks I continue to see the importance of the dot product within the AI space. It has proven to be fully essential. With that there has also been a large amount of emphasis on parallizing these operations through matrix multiplication which I found extremely hard to visualize. The most important thing is that the two leftmost inner dimensions match up. In order to do this you must have a full understanding of the shape of each of these matricies. Putting a lot of effort into understanding the exact shape of each matrix has greatly helped with my ability to both understand and code these values. Knowing the shapes is also extremely important for catching bugs since it's the simplest assertion which can be made and will catch a wide variety of issues within the pipeline.

    - Learning more about the softmax function which is the go to function for doing V-way classification since it converts scores into a scale from 0 to 1 which sums to 1. The softmax is shift invarient which means adding or subtracting a constant will have no affect of the results. This is because the softmax function compares distances between values instead of ratios. This is the reason why we are able to guard against overflow within the softmax by subtracting by the maximum value without any affect on the results. Because the softmax looks at distances between numbers multiplication or division will have a significant affect on the results. This is where temperature values come in by performing scaling you can significantly affect the softmax since it has clear saturation points. This is the reason why we divided the results of Q @ K^t which attempts to decrease the std of the scores to 1 which allows the softmax to perform well.

    - Attention weights help you create hypothesis but ablation is the only way to produce an answer. Many times to much attention is placed on the weights which can be extremely misleading especially in a model with many layers. When looking at the weights it can feel productive and make you feel like you can explain why a model is producing specific results but this isn't the case. The best way to understand what is happening within the attention layer is by making specific changes to the prompt structure by moving peices of the prompt around or removing them and seeing how the results change. If you are worried that the model isn't using specfic facts provided within the prompt inspecting the weights of those tokens isn't likely to be much help. Instead switching the fact with something else and seeing how the results change will leard to a much cleaner answer and will save you a lot of time.

    - Throughout my learning I have placed a large emphasis on how models can be optimized to save cost and the attention layer seems like an easy place since it scales quadriatically with the token sequence. The truth is that under a specific sequence threshold the MLP cacluations dominate the cost not attention. When below this threshold no amount of pruning is going to provide the cost reduction you are looking for. Also when using an API you don't have the ability to prune heads so the emphasis then moves to efficient prompting where you are providing the LLM with only the information it needs. So instead of sending the LLM the full chat history instead only sending information which is relevant to the users newest query is likely to have the largest affect on cost no matter if you are using local models or API's. This is where both prompt caching and KV caches come and are topics which I will learn much more about later on in the curriculum.

## Questions

    - My first question is around prompy/history optimization. Since the sequence length is the most important peice in most architectures for decreasing lost it seems like some sort of RAG/other retreival is extremely important here. Since we only care about providing the model with information it needs to generate the best response there is a lot of fluff which can be removed and retreival to select only relevant information.

    - What are the industry standards when it comes to understanding how the model is being used in production. I feel like storing user prompts could be very important within these applications to not only understand the use cases but also for fine-tuning later on. If you are first creating the application instead of making changes to an existing one I feel like it can be pretty challenging to know which strings to pull because the users might specify one use case but then use it for something entirely different. So is it standard to deploy a more complex model then make decisions later on about cost once you have actual data to back decisions?

    - The way I currently see it is that many companies are using extremely complex models which generalize to many different domains for a narrow amount of tasks. This to me is like hiring a genius and then having them sift through your emails for you a provide daily summaries. You are paying to employ the genius but not making proper use of why you had to pay him so much. Instead you can hire someone who is only good at going through emails and providing summaries which will be much more cost affective and I feel like this is exactly what is happening with language models currently.
