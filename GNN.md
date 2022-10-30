### Stability and Generalization Capabilities of Message Passing Graph Neural Networks

<https://www.arxiv.org/abs/2202.00645>

In graph classification where graph is sampled from different random graph model, derive a non-asymptotic bound on the generalization gap between the empirical and statistical loss, which decreases to zero as the graphs become larger. 

### Constant Time Graph Neural Networks

<https://www.arxiv.org/abs/1901.07868>

GNNs approximate huge graph by sampling the node, and this paper proves whether the query complexity for node sampling is constant time, for different activation, architecture, and forward/backward.

### Minimal Variance Sampling with Provable Guarantees for Fast Training of Graph Neural Networks

<https://www.arxiv.org/abs/2006.13866>

Theoretically analyze the variance of sampling methods and show that, due to the composite structure of empirical risk, the variance of any sampling method can be decomposed into embedding approximation variance in the forward stage and stochastic gradient variance in the backward stage. Propose a decoupled variance reduction strategy.

### Analyzing the expressive power of graph neural networks in a spectral perspective

<https://www.researchgate.net/publication/349119879_ANALYZING_THE_EXPRESSIVE_POWER_OF_GRAPH_NEURAL_NETWORKS_IN_A_SPECTRAL_PERSPECTIVE>

By bridging the gap between the spectral and spatial design of graph convolutions, theoretically demonstrate some equivalence of the graph convolution process regardless it is designed in the spatial or the spectral domain.

### Asymptotics of l2 Regularized Network Embeddings

<https://www.arxiv.org/abs/2201.01689>

Study effects of regularization on embedding in unsupervised random walk, and prove that under exchangeability assumption on the graphs, it leads to learning a nuclear-norm type penalized graphon. In particular, the exact form of penalty depends on the choice of subsampling method used.

### Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks

<https://www.arxiv.org/abs/1810.02244>

Relate Graph Neural Network to 1-dimensional Weisfeiler-Leman graph isomorphism heuristics, show that GNNs have the same expressiveness as the 1-WL in terms of distinguishing non-isomorphic graphs, with same shortcomings. Propose a generalization of GNNs so-called k-dimensional GNNs that can take higher-order graph structures.

### How Powerful are Graph Neural Networks?

<https://www.arxiv.org/abs/1810.00826>

Characterize the discriminative power of GNN variants like Graph Convolution Networks or GraphSAGE, and show that they cannot distinguish certain simple graph structures, and develop provably most expressive architecture, which is as powerful as the Weisfeiler-Lehman graph isomorphism test.

### The Surprising Power of Graph Neural Networks with Random Node Initialization

<https://www.arxiv.org/abs/2010.01179>

Analyze the expressive power of GNNs with Random Node Initialization, prove that these models are universal.

### Transferability Properties of Graph Neural Networks

<https://www.arxiv.org/abs/2112.04629>

For the setting of training graph on moderate size and testing on large graphs, use the graph limit of graphons, and define graph filter and graphon filter to formulate graph/graphon convolution neural network. Using this formulation, bound the error of transferring in same graphon. Show that tranference error decreases with graph size, and graph filters have a transferability-discriminiability tradeoff.

### Graph Neural Networks Are More Powerful Than we Think

<https://www.arxiv.org/abs/2205.09801>

Despite the previous result on limitation of expressivity of GNN by WL algorithm, but alternatively show that this is only the case when the input vector is the vector of all ones. Rather, show that GNNs can distinguish between any graphs that differ in at least one eigenvalue.

### Generalization Analysis of Message Passing Neural Networks on Large Random Graphs

<https://www.arxiv.org/abs/2202.00645>

Show that when training a MPNN on a dataset from random graph models, the generalization gap increases in the complexity of the MPNN, and decreases by the number of samples and average number of nodes. 

### Graph Neural Network Sensitivity Under Probabilistic Error Model

<https://www.arxiv.org/abs/2203.07831>

Study the effect of a probabilistic graph error model on the performance of GCNs. Prove that the adjacency matrix under the error model is bounded by a function of graph size and error probability.

### Implicit Bias of Linear Equivariant Networks

<https://arxiv.org/abs/2110.06084>

Show that L layer full width linear GCNNs trained via gradient descent in a binary classification task converge to solutions with low-rank Fourier matrix coefficients, regularized by the 2/L-Schatten matrix norm. This generalizes previous analysis on the implicit bias of linear CNNs to linear GCNNs over all finite groups, including the challenging setting of non-commutative symmetry groups.

### We Cannot Guarantee Safety: The Undecidability of Graph Neural Network Verification

<https://arxiv.org/abs/2206.05070>

Show that the graph classifier verification is undecidable, however the node classification is verifiable when degree of graph is restricted.

### Lower and Upper Bounds for Numbers of Linear Regions of Graph Convolutional Networks

<https://arxiv.org/abs/2206.00228>

Present the estimates for the number of linear regions of the GCNs, particularlay the optimal upper bound for one-layer GCN and both bounds for multi-layer case, where multi-layer has exponentially many regions than one-layer.

### How does Heterophily Impact Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications

<https://www.arxiv.org/abs/2106.07767>

Show that for homophilous graph data, impactful structural attacks always lead to reduced homophily, while for heterophilous graph data the change in the homophily level depends on the node degree.

### Stability of Aggregation Graph Neural Networks

<https://www.arxiv.org/abs/2207.03678>

Study the stability properties of aggregation graph neural networks considering perturbations of the underlying graph. Derive the stability bound, which is defined by the properties of the filters in the first layers of CNN.

### Understanding the Dynamics of DNNs Using Graph Modularity

<https://www.arxiv.org/abs/2111.12485>

Experimentally observe that the modularity, which measures the evolution of communities, roughly tends to increase as the layer goes deeper and the degradation and plateau arise when the model complexity is great relative to the dataset. Through an asymptotic analysis, prove that modularity can be broadly used for different applications.

### On the expressive power of message-passing neural networks as global feature map transformers

<https://www.arxiv.org/abs/2203.09555>

Investigate the power of MPNN, by using a simple language for feature map transformer, which can express every MPNN. Then give the condition for converse inclusion for exact and approximate expressiveness, or the use of arbitrary activation functions.

### Effects of Graph Convolutions in Multi-layer Networks

<https://www.arxiv.org/abs/2204.09297>

Study the effect of graph convolution for node classification with non-linearly separble Gaussian mixture model. Show that a single graph convolution expands the regime of the distance between the means where multi-layer network can classify the data, by a factor of at least E\[deg\]^(-1/4), and with a slightly stronger graph density, this factor is improced to n^(-1/4), where n is the number of nodes in the graph.

### Rethinking Graph Neural Networks for the Graph Coloring Problem

<https://www.arxiv.org/abs/2208.06975>

For aggregation-combine GNNs, define the power of GNNs in the coloring problem as the capability to assign nodes different colors, and identify node pairs that AC-GNNs fail to discriminate. Also show that AC-GNN is a local coloring method, which becomes non-optimal in the limits over sparse random graphs. Finally prove the positive correlation between model depth and its coloring power.

### How Powerful are K-hop Message Passing Graph Neural Networks

<https://www.arxiv.org/abs/2205.13328>

Show that the expressive power of K-hop message passing is more powerful than 1-hop message passing, however is still impossible to distinguish some simple regular graphs. Define the KP-GNN which also includes the peripheral subgraph information, and prove that it can distinguish almost all regular graphs.

### How Powerful is Implicit Denoising in Graph Neural Networks

<https://www.arxiv.org/abs/2209.14514>

It is believed that GNNs implicitly remove the non-predictive noise, and this work analyze when and why this happens in GNN, by studying the convergence property of noise matrix. Suggest this denoising largely depends on the connextivity, graph size, and GNN architecture. Define adversarial graph signal denoising, and derive a robust graph convolution.

### Tree Mover's Distance: Bridging Graph Metrics and Stability of Graph Neural Networks

<https://www.arxiv.org/abs/2210.01906>

Define a pseudometric for attributed graphs, the Tree Mover's distance, and study its realation to the generalization. Via a hierarachical optimal transport, TMD reflects both local distribution of node attribute and the distribution of local computation tree.

### Graph Neural Networks as Gradient Flows: understanding graph convolutions via energy

<https://www.arxiv.org/abs/2206.10991>

Derive GNNs as a gradient flow, and show that the positive/negative eigenvalues of the channel mixing matrix correspond to attractive/replusive forces. Rigorously prove how the channel mixing can learn to steer the dynamics towards low or high frequencies.

### Graph Neural Networks are Dynamic Programmers

<https://www.arxiv.org/abs/2203.15544>

Using the methods from category thoery, show that there is an intricate connection between GNNs and DP. With this connection, find several prior findings for the algorithmically aligned GNNs.

### On Representing Mixed-Integer Linear Program by Graph Neural Networks

<https://arxiv.org/abs/2210.10759>

Show that there are both feasible and infeasible MILPs that all GNNs treat equally, which indicate that GNN lacks to express general MILP. However show that restricting the MILPs to unfoldable ones or adding random features, there exist GNNs that can predict MILP feasibility, optimal objective value, and optimal solution.

### Superiority of GNN over NN in generalizing bandlimited functions

<https://arxiv.org/abs/2206.05904>

Show that GNN architecture outperform the NN in approximating bandlimited functions on compact d-dimensional Euclidean grids, which only uses few sampled functional values.

### Boosting the Cycle Counting Power of Graph Neural Networks with I^2-GNNs

<https://arxiv.org/abs/2210.13978>

Show that subgraph MPNN cannot count more-than-4-cycles at node level, meaning that the node representation cannot encode the structure like ring with four atomes. Propose I^2-GNN which extend the Subgraph MPNN, that has stronger discriminative power.