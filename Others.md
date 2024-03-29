## Neural Collapse

### Exploring Deep Neural Networks via Layer-Peeled Model: Minority Collapse in Imbalanced Training

<https://www.arxiv.org/abs/2101.12699>

Introduce Layer-Peeled Model which is a nonconvex yet analytically tractable optimization problem, that can better understand deep neural newtorks, obtained by sisolating the topmost layer from the remainder of the neural networks, with some constraints on the two parts of the network. Using this, prove that in class-balanced datasets, any solution forms a simplex equiangular tight frame, and show neural collapse in imbalanced problem.

### An Unconstrained Layer-Peeled Perspective on Neural Collapse

<https://www.arxiv.org/abs/2110.02796>

Prove that gradient flow on unconstrained layer-peeled model converges to critical points of a minimum-norm separation problem exhibiting neural collapse in its global minimizer. Then prove that all the critical points are strict saddle points except the global minimizers that exhibit the neural collapse phenomenon.

### Neural Collapse Under MSE Loss: Proximity to and Dynamics on the Central Path

<https://www.arxiv.org/abs/2106.02073>

Neural Collapse behavior, that last layer features collapse to class-mean, also happens in MSE loss, which is easier to analyze compared to CE loss. Using this, decompose MSE loss to two terms, where one term is directly related to NC. Using this, introduce the central path where the linear classifier stays MSE-optimal for feature, and study the renormalized gradient flow about the central path, and derive the exact dynamics predicting NC.

### An Unconstrained Layer-Peeled Perspective on Neural Collapse

<https://www.arxiv.org/abs/2110.02796>

Introduce a surrogate model called the uncontrained layer-peeled model, and prove that gradient flow on this model converges to critical points of a minimum norm separation problem exhibiting neural collapse. Show that this model with cross-entropy loss has a benign global landscape, allowing to prove that all the critical points are strict saddle points except the global minimizers exhibiting the neural collpase.

### On the Implicit Bias Towards Minimal Depth of Deep Neural Networks

<https://www.arxiv.org/abs/2202.09028>

Study the implicit bias of SGD to favor low-depth solutions when training deep neural networks. Empirically found that neural collapse appears even in intermediate layers, and strengthens when increasing the number of layers, which is evidence of low-depth bias. Characterize notion of effective depth by measuring the minimial layer enjoying neural collapse, and show that effective depth monotonically increases when training with extended portions of random labels.

### Imbalance Trouble: Revisiting Neural-Collapse Geometry

<https://www.arxiv.org/abs/2208.05512>

Adopting the unconstrained-features model which is a recent theoretical model for studying neural collapse, and introduce Simplex-Encoded-Labels Interpolation as characterization of the neural collapse phenomenon. Prove that for the UFM with cross-entropy and vanishing regularization, the embeddings and classifiers always interpolate simplex-encoded label matrix and their individual geometrices are determinied by the SVD factors.

### Are All Losses Created Equal: A Neural Collapse Perspective

<https://www.arxiv.org/abs/2210.02192>

Extend the results on the neural collapse, show that label smoothing, focal loss also exhibit neural collapse and give equivalent feature. Show that the only global minimizers are neural collapse solutions.

### Neural Collapse in Deep Linear Network: From Balanced to Imbalanced Data

<https://www.arxiv.org/abs/2301.00437>

Prove that the neural collapse occur in deep linear network for MSE and CE loss, and extend this setting to imbalanced data and present the geometric analysis.

### Generalizing and Decoupling Neural Collapse via Hyperspherical Uniformity Gap

<https://www.arxiv.org/abs/2303.06484>

Propose the generalized neural collapse hypothesis, which removes the assumption on the feature dimension and the number of classes. Derive new objective named hyperspherical uniformity gap, which converges to the generalized cneural collapse and have more flexity.

### Are Neurons Actually Collapsed? On the Fine-Grained Structure in Neural Representations

<https://www.arxiv.org/abs/2306.17105>

Empirically find that even after collapse, small variations are enough to capture the input distribution's structure. Theoretically show the fine-grained representation structure in simple synthetic setting.

### A Study of Neural Collapse Phenomenon: Grassmannian Frame, Symmetry and Generalization

<https://www.arxiv.org/abs/2304.08914>

Obtain the Grassmannian frame structure of optimization and generalization, which maximally separates features of two classes while not requiring larger feature dimension than number of classes. Show the symmetric generalization of this Grassmannian frame of permutation.

### Neural (Tangent Kernel) Collapse

<https://www.arxiv.org/abs/2305.16427>

Under the assumption that empirical NTK has block structure aligned with class labels, derive the dynamics of DNNs and find the invariant which proves emergence of neural collapse.

## Hierarchical Tensor Decomposition

### Analysis and Design of Convolutional Networks via Hierarchical Tensor Decompositions

<https://www.arxiv.org/abs/1705.02302>

Through an equivalence to hierarchical tensor decompositions, analyze the expressive efficiency and inductive bias of various convolutional network architectural features.

### Implicit Regularization in Hierarchical Tensor Factorization and Deep Convolutional Neural Networks

<https://www.arxiv.org/abs/2201.11729>

By dynamical system approach, analyzes the implicit regularization in hierarchical tensor factorization, establish the implicit regularization towards low rank, which translates to locality.

### Implicit Regularization in Hierarchical Tensor Factorization and Deep Convolutional Neural Networks

<https://www.arxiv.org/abs/2201.11729>

Theoretically analyze the implicit regularization in hierarchical tensor factorization which is a model equivalent to certain deep CNNs, and establish implicit regularization towards low hierarchical tensor rank. This translates to an implicit regularization towards locality for the associated convolutional networks.

## Memorization

### On the Optimal Memorization Power of ReLU Neural Networks

<https://www.arxiv.org/abs/2110.03187>

Show that networks can memorize any N points using sqrt(N) parameters with some separability assumptions, which is optimal up to logarithmic factors.

## Pruning

### Spectral Pruning for Recurrent Neural Networks

<https://www.arxiv.org/abs/2105.10832>

Propose a pruning algorithm so called spectral pruning for RNN, and provide the generalization error bounds for compressed RNNs.

### Why Lottery Ticket Wins? A Theoretical Perspective of Sample Complexity on Pruned Neural Networks

<https://www.arxiv.org/abs/2110.05667>

Characterizes the performance of training a pruned neural network by analyzing the geometric structure of the objective function and the sample complexity to achieve zero generalization error. Show that the convex region near a desirable model with guaranteed generalization enlarges as the neural network model is pruned. 

### Analyzing Lottery ticket Hypothesis from PAC-Bayesian Theory Perspective

<https://www.arxiv.org/abs/2205.07320>

Hypothesize that the 'winning tickets' have relatively sharp minima, which is a disadvantage in terms of generalization ability, and confirm this hypothesis with PAC-Bayesian theory. Find that the flatness is useful for improving the accuracy and robustness to label noise, and the distance from the initial weights is deeply involved in winning tickets.

### Theoretical Characterization of How Neural Network Pruning Affects its Generalization

<https://www.arxiv.org/abs/2301.00335>

Study how different pruning fractions affect the model's gradient descent dynamics and generalization. Show that for two-laer NN, as long as the pruning fraction is below a certain threhold, the neural network still gives zero training loss with good generalization error. Moreover, the generalization bound is better with large pruning fraction. However there exists large pruning fraction that gradient descent gives zero training loss but generalization similar to random guessing.

### Most Activation Functions Can Win the Lottery Without Excessive Depth

<https://www.arxiv.org/abs/2205.02321>

Show that the target network with depth L can be approximated by the subnetwork of a randomly initialized network with depth L+1, and wider by a logarithmic factor. This analysis extends to large class of activation functions, those can be asymptotically approximated by LeakyReLU.

### Polynomially Over-Parameterized Convolutional Neural Networks Contain Structured Strong Winning Lottery Tickets

<https://arxiv.org/abs/2311.09858>

Show the wide class of randomly ininitialised CNNs contain structured subnetworks, that can approximate any sufficiently smaller network. 

## Other training

### Global Optimality Beyond Two Layers: Training Deep ReLU Networks via Convex Programs

<https://www.arxiv.org/abs/2110.05518>

Show that the training of multiple three-layer ReLU sub-networks with weight decay regularization can be equivalently cast as a convex optimization problem in a higher dimensional space, where sparsity is enforced via a group l1-norm regularization. Then prove that equivalent convex problem can be globally optimized by a standard convex optimization solve with a polynomial-time complexity w.r.t. number of samples and data dimension.

### Convergence Analysis and Implicit Regularization of Feedback Alignment for Deep Linear Networks

<https://www.arxiv.org/abs/2110.10815>

Provide convergence guarantees with rates for deep linear networks for both continuous and discrete dynamics on FA algorithms.

### Faster Neural Network Training with Approximate Tensor Operations

<https://www.arxiv.org/abs/1805.08079>

Introduce a new technique for faster NN training using sample-based approximation to the tensor operations, prove that they provide the same convergence guarantees.

### Depth Without the Magic: Inductive Bias of Natural Gradient Descent

<https://www.arxiv.org/abs/2111.11542>

Gradient descent has implicit inductive bias, that the parameterization gives different optimization trajectory. Natural gradient descent is approximately invariant to such parameterization, giving same trajectory and same minimum. Show that there exist learning problem where natural gradient descent fails to generalize while gradient descent performs well.

### Provably Training Overparameterized Neural Network Classifiers with Non-convex Constraints

<https://www.arxiv.org/abs/2012.15274>

Show that overparameterized neural networks could achieve a near-optimal and near-feasible solution of non-convex constrained optimization via the projected SGD, using the no-regret analysis of online learning.

### How Does Sharpness-Aware minimization Minimize Sharpness?

<https://www.arxiv.org/abs/2211.05729>

SAM generally minimizes the two step approximation of sharpness which is computationally efficient, so study whether this difference creates difference in empirical results. Clarify the exact sharpness notion that SAM regularizes, and show that the two step approximations individually lead inaccurate local conclusion, but gives correct effect combined.

## implicit Regularization

### The Equilibrium Hypothesis: Rethinking implicit regularization in Deep Neural Networks

<https://www.arxiv.org/abs/2110.11749>

Recent work showed that some layers are much more aligned with data labels than other layers, called impricial layer selection. Introduce and empirically validate the Equilibrium Hypothesis stating that the layers achieve some balance between forward and backward information loss are the ones with the highest alignment to data labels.

### The Geometric Occam's Razor Implicit in Deep Learning

<https://www.arxiv.org/abs/2111.15090>

Over-parameterized neural networks trained with SGD are subject to a Geometric Occam's Razor, that they are implicitly regularized by the geometric model complexity, which is a Dirichlet energy of the function.

### Implicit Regularization Towards Rank Minimization in ReLU Networks

<https://www.arxiv.org/abs/2201.12760>

Prove that GF on ReLU networks may no longer tend to minimize ranks, while revealing that ReLU networks of sufficient depth are provably biased towards low-rank solution.

## Double Descent

### Phenomenology of Double Descent in Finite-Width Neural Networks

<https://www.arxiv.org/abs/2203.07337>

Study the population loss with its lower bound using influence functions, which connects the spectrum of the Hessian at the optimum, and exhibit a double descent behaviour at the interpolation threshold.

## Linear Neural Network

### The Power of Contrast for Feature Learning: A Theoretical Analysis

<https://www.arxiv.org/abs/2110.02473>

By using connection between PCA and linear Autoencoder, GAN, contrast learning, show that contrastive learning outperforms autoender for both feature learning and downstream tasks.

### Deep Contrastive Learning is Provably (almost) Principal Component Analysis

<https://www.arxiv.org/abs/2201.12680>

Show that contrastive learning has a game-theoretical formulation, where max-player maximizes contrastiveness, min-player puts weights on pairs of samples with similar representation. Show that max player reduces to PCA for deep linear networks, with all local minima as global minima. This is also extended to 2-layer ReLU networks, and prove that feature composition is preferred then single dominant feature under strong augmentation.

### Principal Components Bias in Over-parameterized Linear Models, and its Manifestation in Deep Neural Networks

<https://www.arxiv.org/abs/2105.05553>

In over-parameterized deep linear network with enough width, the convergence rate of parameters is exponentially fast along the larger principal components of the data, with rate governed by the singular value, named principal-component bias. Discuss how this may explain benefits of early stopping and why deep networks converge slowly with random labels.

### A spectral-based analysis of the separation between two-layer neural networks and linear methods

<https://www.arxiv.org/abs/2108.04964>

Propose a spectral based approach to analyze how two-layer neural networks separate from linear methods. This can be reduced to estimating the Kolmogorov width of two-layer neural networks, which can be characterized using the spectrum of an associated kernel. This allows upper bound, lower bound, and identifying explicit hard functions, and systematic study of choice of activation's effect on the separation.

### Exact Solutions of a Deep Linear Network

<https://www.arxiv.org/abs/2204.04777>

Find the analytical expression of global minima of deep linear networks with weight decay and stochastic neurons. Show that weight decay can create bad minima at zero, and that most of the initialisation are insufficient.

### (S)GD over Diagonal Linear Networks: Implicit bias, Large Stepsizes and Edge of Stability

<https://arxiv.org/abs/2302.08982>

Show that GD and SGD with macroscopic step sizes in overparameterised regime converges and derive the solution through implicit regularisation. Show that large step size benefit SGD for sparse regression, while it is detrimental for GD, and such behaviours are magnified at the edge of stability.

## Others

### Why resampling outperforms reweighting for correcting sampling bias with stochastic gradients

<https://www.arxiv.org/abs/2009.13447>

Explain the reason that resampling outperforms reweighting using tools from dynamical stability and stochastic asymptotics.

### Emergence of memory manifolds

<https://www.arxiv.org/abs/2109.03879>

Present a general principle called frozen stabilisation, allowing a family of neural networks to self-organise to a critical state exhibiting memory manifolds without parameter fine-tuning or symmetries.

### A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning

<https://www.arxiv.org/abs/2109.02355>

Provides a succinct overview of this emerging theory of overparameterized ML that explains recent findings through a statistical signal processing perspective.

## Learning Dynamics of Deep Networks Admit Low-rank Tensor Descriptions

<https://openreview.net/pdf?id=Hy7RHt1vz>

Propose a simple tensor decomposition model to study how hidden representations evolve over learning, which precisely extracts the correct dynamics of learning and closed form solutions.

### Understanding Black-box Predictions via Influence Functions

<https://www.arxiv.org/abs/1703.04730>

Show that even on non-convex and non-differentiable models, approximations to influence functions can still provide valuable information. 

### Understanding Convolutional Neural Networks with Information Theory: An Initial Exploration

<https://www.arxiv.org/abs/1804.06537>

Show that the estimators enable straightforward measurement of information flow in realistic convolutional neural networks without any approximation, and introduce the partial information decomposition framework, develop three quantities to analyze the synergy and redundancy in convolutional layer representations.

### Searching for Minimal Optimal Neural Networks

<https://www.arxiv.org/abs/2109.13061>

Propose a rigorous mathematical framework for studying the asymptotic theory of the destructive technique, and prove that Adaptive group Lasso is consistent and can reconstruct the correct number of hidden nodes of one-hidden-layer feedforward networks with high probability.

### On the Variance of the Fisher Information for Deep Learning

<https://www.arxiv.org/abs/2107.04205>

Investigate two estimators based on two equivalent representations of the FIM, and bound their variances and analyze how the parametric structure of a deep neural network can impact the variance.

### Avoiding pathologies in very deep networks

<https://www.arxiv.org/abs/1402.5836>

Show that in standard architectures, the representational capacity of the network tends to capture fewer degrees of freedom as the number of layers increases, and propose an alternate architecture which does not suffer from this pathology.

### Optimizing Neural Networks via Koopman Operator Theory

<https://www.arxiv.org/abs/2006.02361>

Show that Koopman operator theoretic methods allow for accurate predictions of weights and biases of MLPs over a non-trivial range of training time.

### Stability of Neural Networks on Manifold to Relative Perturbations

<https://www.arxiv.org/abs/2110.04702>

Prove that manifold neural networks composed of frequency ratio threshold filters, which separates the infinite-dimensional spectrum of the Laplace-Beltrami operator, are stable to relative operator perturbations. Observe that manifold neural networks exhibit a trade-off between stability and discriminability.

### Phase Collapse in Neural Networks

<https://www.arxiv.org/abs/2110.05283>

By defining simplified complex-valued convolutional network architecture, which implements convolution with wavelet filters and uses a complex modulus to collapse phase variables, demonstrate that it is a different phase collapse mechanism which explains the ability to progressively eliminate spatial variability.

### Does Preprocessing Help Training Over-parameterized Neural Networks?

<https://www.arxiv.org/abs/2110.04622>

Design preprocessing algorithm for layer and input data, with convergence guarantee and lower train cost.

### Understanding Learning Dynamics of Binary Neural Networks via Information Bottleneck

<https://www.arxiv.org/abs/2006.07522>

Analyze BNNs through the information bottleneck principle and observe that the training dynamics of BNNs is different from that of DNNs. While DNNs have a separate empirical risk minimization and representation compression phases, BNNs tend to find efficient hidden representations concurrently with label fitting.

### Well-classified Examples are Underestimated in Classification with Deep Neural Networks

<https://www.arxiv.org/abs/2110.06537>

Theoretically show that giving less gradient for well-classified examples hinders representation learning, energy optimization, and the growth of margin. Propose to reward well-classified examples with additive bonuses to revive their contribution to learning.

### Detecting Modularity in Deep Neural Networks

<https://www.arxiv.org/abs/2110.08058>

Consider the problem of assessing the modularity exhibited by a partitioning of a network's neurons. Propose two proxies, importance and coherence measured by statistical methods. Then apply the proxies to partitionings generated by spectrally clustering neurons and show that these partitionings reveal groups of neurons that are important and coherent.

### Dropout as a Regularizer of Interaction Effects

<https://www.arxiv.org/abs/2007.00823>

Prove that dropout regularizes against higher-order interactions. 

### Understanding Convolutional Neural Networks from Theoretical Perspective via Volterra Convolution

<https://www.arxiv.org/abs/2110.09902>

Show that CNN is an approximation of the finite term Volterra convolution, whose order increases exponentially with the number of layers and kernel size increases exponentially with the strides.

### Expressivity of Neural Networks via Chaotic Itineraries beyond Sharkovsky's Theorem

<https://www.arxiv.org/abs/2110.10295>

Prove that periodic points alone lead to suboptimal depth-width tradeoffs and improve upon them by demonstrating that certain "chaotic itineraries" give stronger exponential tradeoffs. Identify a phase transition to the chaotic regime that exactly coincides with an abrupt shift in other notions of function complexity, including VC-dimension and topological entropy.

### Early Stopping in Deep Networks: Double Descent and How to Eliminate It

<https://www.arxiv.org/abs/2007.10099>

Show that epoch-wise double descent arises by a superposition of two or more bias-variance tradeoff that arise because different parts of the network are learned at different epochs, and eliminating this by proper scaling of stepsizes can significantly improve the early stopping performance. Show this analytically for linear regression and a two-layer neural network.

### Wide Neural Networks Forget Less Catastrophically

<https://www.arxiv.org/abs/2110.11526>

Focus on the model and study the impact of width of the NN architecture on catastrophic forgetting, and show that width has a suprisingly significant effect. Study the learning dynamics of the network from various perspectives, including gradient norm, sparsity, orthogonalization, lazy training.

### Does the Data Induce Capacity Control in Deep Learning?

<https://www.arxiv.org/abs/2110.14163>

Show that the data correlation matrix, Hessian, Fisher Information Matrix all share 'sloppy' eigenspectrum where a large number of small eigenvalues are distributed uniformly over an exponentially large range. Show that this structure in the data can give to non-vacuous PAC-Bayes generalization bounds analytically.

### What training reveals about neural network complexity

<https://www.arxiv.org/abs/2106.04186>

Explores Benevolent Training Hypothesis, that the complexity of target function can be deduced by training dynamics. Observe that the Lipscitz constant close to the training data affects various aspects of the parameter trajectory, with more complex network having longer trajectory, bigger variance. Show that NNs whose first layer bias is trained more steadily have bounded complexity, and find that steady training with dropout implies a training and data-dependent generalization bound growing poly-logarithmically with the number of parameters. 

### Collapse of Deep and Narrow Neural Nets

<https://www.arxiv.org/abs/1808.04947>

Show that even for ReLU activation, deep and narrow NNs will converge to errorneous mean or median states of the target function depending on the loss with high probability. 

### Why Stable Learning Works? A Theory of Covariate Shift Generalization

<https://www.arxiv.org/abs/2111.02355>

Prove that under ideal conditions, stable learning algorithms could identify minimal stable variable set, that is minimal and optimal to deal with covariate shift generalization for common loss functions.

### Early-stopped neural networks are consistent

<https://www.arxiv.org/abs/2106.05932>

Show that gradient descent with early stopping achieves population risk arbitrarily close to optimal in terms of not just logistic and misclassification losses, but also in terms of calibration.

### Multiple Descent: Design Your Own Generalization Curve

<https://www.arxiv.org/abs/2008.01036>

Show that the generalization curve can have an arbitrary number of peaks, and the locations of those peaks can be explicitly controlled in variable parameterized families of models on linear regression. The emergence of double descnet is due to the interaction between the properties of the data and the inductive bias of learnin algorithms.

### Stability & Generalisation of Gradient Descent for Shallow Neural Networks without the Neural Tangent Kernel

<https://www.arxiv.org/abs/2107.12723>

Show oracle type bounds which reveal that the generalisation and excess risk of GD is controlle by an interpolating network with the shortest GD path from inistialisation. Also show that this analysis is tighter then existing NTK-based risk bounds, and show that GD with early stoppping is constant.

### ReLU Neural Networks of Polynomial Size for Exact Maximum Flow Computation

<https://www.arxiv.org/abs/2102.06635>

Introduce the concept of Max-Affine Arithmetic Programs, and use them to show that undirected graph's minimum spanning tree and maximum flow computation is possible with NNs of cubic/quadratic width.

### Assessing Deep Neural Networks as Probability Estimators

<https://www.arxiv.org/abs/2111.08239>

Find that the likelihood probability density and the inter-categorical sparsity have greater impacts than the prior probability to DNN's classification uncertainty.

### Towards Understanding the Condensation of Neural Networks at Initial Training

<https://www.arxiv.org/abs/2105.11686>

Empirically, it is observed that input weights condense on isolated orientation with a small initialization. Show that maximal number of condensed orientation in the initial stage is twice the multiplicity of the acitvation function, where multiplicity is multiple roots of activation function at origin. 

### Improved Fine-tuning by Leveraging Pre-training Data: Theory and Practice

<https://www.arxiv.org/abs/2111.12292>

Show that final prediction precision may have a weak dependency on the pre-trained model especially in the case of large training terations. Shows that the final performance can be improved when appropriate pre-training data is included in fine-tuning, and design a novel selection strategy to select a subset from pre-training data to help improve the generalization.

### Gradient Starvation: A Learning Proclivity in Neural Networks

<https://www.arxiv.org/abs/2011.09468>

Gradient Starvation arises when cross-entropy loss is minimized by capturing only a subset of features relevant for the task, despite the presence of other predictive features that fail to be discovered. Using tools from dynamical systems theory, identify simple properties of learning dynamics during gradient descent that lead to this imbalance, and prove that such a situation can be expected given certain statistical structure in training data.

### Error Bounds for a Matrix-Vector Product Approximation with Deep ReLU Neural Networks

<https://www.arxiv.org/abs/2111.12963>

Derive error bounds in Lebesgue and Sobolev norms to approximate arbitrary matrix-vector product using ReLU NN.

### On the rate of convergence of a classifier based on a Transformer encoder

<https://www.arxiv.org/abs/2111.14574>

The rate of convergence of the misclassification probability towards the optimal misclassification probability, and shown that this classifier is able to circumvent the curse of dimensionality.

### Breaking the Convergence Barrier: Optimization via Fixed-Time Convergent Flows

<https://www.arxiv.org/abs/2112.01363>

Design gradient based optimization for achieving acceleration, by first leveraging a continuous time framework for designing fixed-time stable dynamical systems, provigding a consistent discretization strategy such that the equiavlent discrete-time algorithm tracks the optimizer in a practically fixed number of iterations. Provide the convergence behavior of the proposed gradient flow and robustness to additive disturbances. 

### Asymptotic properties of one-layer artificial neural networks with sparse connectivity

<https://www.arxiv.org/abs/2112.00732>

Asymptotic of empirical distribution of parameters of a one-layer ANNs with sparse connectivity, with increasing number of parameter and iteration steps.

### Test Sample Accuracy Scales with Training Sample Density in Neural Networks

<https://www.arxiv.org/abs/2106.08365>

Propose an error function for piecewise linear NNs taking a local region of input space and smooth empirical training error, that is an average of empirical training erros from other regions weighted by network represenation distance. A bound on the expected smooth error for each region scales inversely with training sample density in representation space.

### On transversality of bent hyperplane arrangements and the topological expressiveness of ReLU neural networks

<https://www.arxiv.org/abs/2008.09052>

Define notion of a generic transversal ReLU neural network, and show that almost all ReLU networks are generic and transversal. Using the obstruction, prove that a decision region of a generic, transversal ReLU network with a single hidden layer of dimension n+1 can have no more than one bounded connected components.

### A Complete Characterisation of ReLU-Invariant Distributions

<https://www.arxiv.org/abs/2112.06532>

Give a complete characterization of families of probability distributions that are invariant under the action of ReLU NN layers, proving that no invariant parameterised can exist unless one of follwoing holds, network's width is one, probability measure have finite support, and the parameterization is not locally Lipscitz continuous.

### On the Expected Complexity of Maxout Networks

<https://www.arxiv.org/abs/2107.00379>

Number of activation regions are used as a complexity measure, and it has shown that practical complexity of Deep ReLU networks is often far from the theoretical maximum. Show that this also occurs in maxout activation, and give nontrivial lower bounds on the complexity, finally gives that different initialization can increase speed of convergence.

### Neurashed: A Phenomenological Model for Imitating Deep Learning Training

<https://www.arxiv.org/abs/2112.09741>

Design a graphical model neurashed, which inherits hierarchically structured, optimized through SGD, and information evolving compressively, and enables insights into implicit regularization, information bottleneck, and local elasticity.

### Effective Sample Size, Dimensionality, and Generalization in Covariate Shift Adaption

<https://www.arxiv.org/abs/2010.01184>

Covariate shift adaption usually suffer from small effective sample size, which is common in high dimensional setting. Focus on unified view connecting ESS, data dimensionality, and generalization in covariate shift adaption, and demonstrate how dimensionality reduction or feature selection increase the ESS. 

### A mean-field optimal control formulation of deep learning

<https://link.springer.com/article/10.1007/s40687-018-0172-y>

Introduces the mathematical formulation of viewing population risk minimization as mean-field optimal control problem, and prove stability condition of the Hamilton-Jacobi-Bellman type and Pontryagin type. By mean-field Pontryagin's maximum principle, establish quantitative relationships between population and empirical learning problem.

### Deep Neural Networks Learn Meta-Structures from Noisy Labels in Semantics Segmentation

<https://www.arxiv.org/abs/2103.11594>

Even with extremely noisy label on semantic segmentation, DNN still provide similar segmenetation performance as trained with original ground truth, ondicating that DNNs learn structures hidden in labels rather than pixel-level labels. Referring this structure as meta-structure, define this formally as spatial density distribution showing both theoretically and experimentally how this explains the behavior.

### An unfeasiability view of neural network learning

<https://www.arxiv.org/abs/2201.00945>

Define notion of a continuously differentiable perfect learning algorithm, and show that such algorithms don't exist given that length of the data set exceeds the number of involved parameters, with logistic, tanh, sin activation.

### The Many Faces of Adversarial Risk

<https://www.arxiv.org/abs/2201.08956>

Make the definition of adversarial risk rigorous, generalize Strassen's theorem to unbalanced optimal transport setting, show the pure Nash equilibrium between adversary and algorithm, and characterize adversarial risk by the minimum Bayes error between a pair of distributions to the infinity Wasserstein uncertainty sets.

### Post-training Quantization for Neural Networks with Provable Guarantees

<https://www.arxiv.org/abs/2201.11113>

Modify GPFQ, a post-trainig NN quantization method based on greedy path following, and prove that for quantizing a single-layer network, the relative square error decys linearly in the number of weights.

### The Implicit Bias of Benign Overfitting

<https://www.arxiv.org/abs/2201.11489>

Show that benign overfitting, where a predictor perfectly fits noisy training data while having low expected loss, is biased towards to certain types of problems, so that it is not general behaviors. In classification setting, prove that the max-margin predictor is asymptotically biased towards minimizing the expected squared hinge loss.

### Critical Initialization of Wide and Deep Neural Networks through Partial Jacobians: General Theory and Applications

<https://www.arxiv.org/abs/2111.12143>

Define criticality by partial Jacobian, which is jacobian between preactivations in different layers, and derive recurrence relation between norms of partial Jacobians with analyzing criticality.

### Fluctuations, Bias, Variance & Ensembles of Learners: Exact Asymptotics for Convex Losses in High-Dimension

<https://www.arxiv.org/abs/2201.13383>

Provide a complete description of the asymptotic joint distribution of the empirical risk minimizers for generic convex loss and regularisation in the high dimensional limit. 

### Interplay between depth of neural networks and locality of target functions

<https://www.arxiv.org/abs/2201.12082>

Introduce k-local and k-global functions, and find that depth is beneficial for learning local functions but detrimental to learning global functions.

### Spectral Analysis and Fixed Point Stability of Deep Neural Dynamics

<https://www.arxiv.org/abs/2011.13492>

Analyze the eigenvalue spectra and stability of discrete-time dynamics systems parameterized by DNNs, viewing neural network as affine parameter varying maps, and analyze using classical system methods. 

### The Implicit Bias of Gradient Descent on Generalized Gated Linear Networks

<https://www.arxiv.org/abs/2202.02649>

Derive infinite-time training limit of a mathematically tractable class of deep nonlinear neural networks, gated linear networks, and generalize to gated networks described by general homogeneous polynomials. Using this, show how architectural constraints and implicit bias affect performance, and that theory captures a substantial portion of the inductive bias of ReLU networks.

### Benign Overfitting in Two-layer Convolutional Neural Networks

<https://www.arxiv.org/abs/2202.06526>

Show that when the signal-to-noise ratio satisfies a certain condition, a two-layer CNN with gradient descent arbitrary small training and test loss, giving benign overfitting. Conversely if this condition does not hold, the CNN only achieve constant level test loss, giving harmful overfitting.

### How and what to learn:The modes of machine learning

<https://www.arxiv.org/abs/2202.13829>

Propose a new approach named weight pathway analysis, which decomposes a neural network into a series of subnetworks of weight pathways. Using WPA, discover that a neural network stores and utilizes information in a holographic way, that the network encodes all training samples in a coherent structure. Also reveal two learning mode of a neural newtwork, linear and nonlinear, where the former extracts linearly separable features, and the latter extracts linearly inseparable features.

### Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data

<https://www.arxiv.org/abs/2010.03622>

Under 'expansion' assumption that low probability subset of data must expand to a neighborhood with large probability relative to the subset, prove that the minimizers of population objectives based on self-training and input-consistency regularization will achieve high accuracy w.r.t. ground-truth labels. Also provide generalization bound and sample complexity guarantee for neural nets.

### Deep Learning meets Nonparameteric Regression: Are Weight-Decayed DNNs Locally Adaptive?

<https://www.arxiv.org/abs/2204.09664>

Using parallel NN variant of ReLU networks, show that the standard weight decay is equivalent to promoting lp-sparsity of the coefficient vector. Using this equivalence, establish that by tuning only the weight decay, such parallel NN achieves an estimation error arbitrarily close to the minimax rates for both Besov and BV classes.

### Shallow Univariate ReLu Networks as Splines: Initialization, Loss Surface, Hessian, & Gradient Flow Dnamics

<https://www.arxiv.org/abs/2008.01772>

Reparameterize the ReLU NN as continuous piecewise linear spline, and study the learning dynamics of univariate ReLU NN with this spline view. Develop a simple view of the structure of the loss surface including critical, fixed points and Hessian. Also show that standard initialization gives very flat function.

### Beyond Folklore: A Scaling Calculus for the Design and Initialization of ReLU Networks

<https://www.arxiv.org/abs/1906.04267>

Propose a system for calculating the scaling constant for layers and weights. Argue that the network is preconditioned by the scaling, an argue that geometric mean of fan-in and fan-out should be used for initialization of the variance of weights.

### Quasi-Equivalence of Width and Depth of Neural Networks

<https://www.arxiv.org/abs/2002.02515>

Formulate two transforms for mapping an arbitrary ReLU network to a wide network and a deep network respectively, for either regression or classification. 

### Rank Diminishing in Deep Neural Networks

<https://www.arxiv.org/abs/2206.06072>

Show that the rank of network monotonically decrease w.r.t. the depth. Also provide the empirical analysis of per-layer bahviour of network rank on ResNet and MLP, Transformers.

### A Theoretical Understanding of Neural Network Compression from Sparse Linear Approximation

<https://www.arxiv.org/abs/2206.05604>

Propose to use sparsity-sensitive lq-norm with 0 < q < 1 to characterize the compressibility and provide a relationship between soft sparsity of the weights and the degree of compression with a controlled accuracy degradation bound.

### SGD Noise and Implicit Low-Rank Bias in Deep Neural Networks

<https://www.arxiv.org/abs/2206.05794>

Prove that when training with weight decay, the only solutions of SGD at convergence are zero functions. Also show that when training with a neural network using SGD with weight decay and small batch size, the resulting weight matrices are expected to be of small rank.

### Local Identifiability of Deep ReLU Neural Networks: the Theory

<https://www.arxiv.org/abs/2206.07424>

Introduce a local parameterization of a deep ReLU NN by fixing the values of some of its weights, which define local lifting operator whose inverses are charts of a smooth manifold of a high dimensional space. The function implemented by the deep ReLU NN composes the local lifting with a linear operator which depends on the sample. Using this representation, derive a necessary and sufficient condition of local identifiability.

### The Role of Depth, Width, and Activation Complexity in the Number of Linear Regions of Neural Networks

<https://www.arxiv.org/abs/2206.08615>

Provide a precise bounds on the maximal number of linear regions of piecewise-linear networks based on depth, width, and activation complexity. Based on combinatorial structure of convex partition, show that the number of regions increase exponentially w.r.t. depth. Also show that along 1D path, the expected density is bounded by the product of depth, width, and a measure of activation complexity, showing the identical role for all three sources.

### Information Geometry of Dropout Training

<https://www.arxiv.org/abs/2206.10936>

Show that the dropout flattens the model manifold, and the regularization performance depends on the curvature. Then show that the dropout corresponds to a regularization that depends on the Fisher information.

### Informed Learning by Wide Neural Networks: Convergence, Generalization and Sampling Complexity

<https://www.arxiv.org/abs/2207.00751>

Study the informed DNN with over-parameterization with domain-knowledge informed to the training objective. Domain knowledge regularizes the label-based supervision and supplements the labeled samples, and this reveals the trade-off between label and knowledge imperfectness in the bound of the population risk. 

### Wide Neural Networks Forget Less Catastrophically

<https://www.arxiv.org/abs/2110.11526>

Show that width has significant effect on forgetting, by studying the learning dynamics of the network from perspectives of gradient orthogonality, sparsity, and lazy training.

### Hidden Progress in Deep Learning: SGD Learns Parities Near the Computational Limit

<https://www.arxiv.org/abs/2207.08799>

Considering the problem of learning k-sparse parities of n bits, observe that empirically, variety of architectures learn with n^O(k) examples with loss suddenly dropping at n^O(k) iterations, and match known sample query lower bound. Then elucidate the mechanism of this phenomena theoretically, that the phase transition is due to gradual amplification of Fourier gap in the population gradient, rather than SGD finding the hidden set of features.

### How Wide Convolutional Neural Networks Learn Hierarchical Tasks

<https://www.arxiv.org/abs/2208.01003>

Study the CNN in kernel regime, show that the spectrum of the kernel and its asymptotic inherit the hierarchical structure of the network. Then use generalization bounds to prove that deep CNNs adapt to the spatial scale of the target function. Illustrate this by computing the convergence rate in a teacher-student setting with teacher randomly initialized, and find that if teacher depends on low-dimensional subset then rate is determined by effective dimensionality, but if teacher depends on all input, then the rate is inversely proportional to the input dimension.

### Improving the Trainability of Deep Neural Networks through Layerwise Batch-Entropy Regularization

<https://www.arxiv.org/abs/2208.01134>

Introduce the batch entropy to analyze the flow of information through neural network, and prove that a positive batch-entropy is required for gradient descent based approaches to optimize given loss function succesfully. Introduce batch entropy regularization based on this insight, showing that we can train deep Vanilla networks with 500 layers without any techniques.

### On the Activation Function Dependence of the Spectral Bias of Neural Networks

<https://www.arxiv.org/abs/2208.04924>

Provide a theoretical explanation for the spectral bias of ReLU neural networks via the connection with the finite element methods, and predict that switching the activation function to a piecewise linear B-spline will remove this spectral bias.

### Sharp asymptotics on the compression of two-layer neural networks

<https://www.arxiv.org/abs/2205.08199>

Consider the setting of learning infinitely wide neural network's output with narrow network, and provide the error rate of this compression.

### A Theory for Knowledge Transfer in Continual Learning

<https://www.arxiv.org/abs/2208.06931>

Present a theory for knowledge transfer in continual supervised learning, for both forward and backward transfer. derive error bounds for each of these transfer, which is agnostic to the architecture.

### How Does Data Freshness Affect Real-time Supervised Learning?

<https://www.arxiv.org/abs/2208.06948>

Show that the performance of real time supervised learning degrades monotonically, if the feature and target data sequence can be closely approximated as a markov chain. The prediction error is a function of the Age of information, where the function could be non-monotonic.

### On the Decision Boundaries of Neural Networks: A Tropical Geometry Perspective

<https://www.arxiv.org/abs/2002.08838>

Using the tropical geometry, show that the decision boundaries of shallow ReLU NN are a subset of a tropical hypersurface, related to a polyope formed by the convex hull of two zonotopes. Propose a tropical perspective to the lottery ticket hypothesis, tropical based reformulation of network pruning, and the generation of adversarial attack.

### Normalization effects on deep neural networks

<https://www.arxiv.org/abs/2209.01018>

Show that using 1/N for the normalization factor, which is known as mean field scaling, is the best choice in terms of the variance of output and test accurcay, with suitable learning rate choices. Also show that this is particularly true for the outer layers, since the behavior is more seneitive to outer layer's scalings.

### Functional dimension of feedforward ReLU neural networks

<https://www.arxiv.org/abs/2209.04036>

ReLU NN's parameter space admits positive-dimensional spaces of symmetries, hence the local functional dimension near any parameter choice is lower than the parameteric dimension. Define the notion of this functional dimension, and show that it is inhomogeneous across the parameter space, and study it by finding when the functional dimension achieves its theoretical maximum, and fibers and quotient spaces.

### Why neural networks find simple solutions: the many regularizers of geometric complexity

<https://www.arxiv.org/abs/2209.13083>

Develop the notion of geometric complexity, that measures variability of the model function using a discrete Dirichlet energy. Show that many training heuristics such as norm regularization. spectral norm regularization, flatness regularization, implicit gradient regularization, noise regularization, and the choices of initialization all act to control this geometric complexity.

### Implicit Bias of Large Depth Networks: a Notion of Rank for Nonlinear Functions

<https://www.arxiv.org/abs/2209.15055>

Show that the representation cost of FCNNs converges to a notion of rank over nonlinear functions, as the depth of the network goes to infinity. Show that too large depths give global minimum to approximately rank 1.

### Overparameterized ReLU Neural Networks Learn the Simplest Models: Neural Isometry and Exact Recovery

<https://www.arxiv.org/abs/2209.15265>

Show that the ReLU networks with an arbitrary numbers of parameters learn only simple models that explain the data, which is analogous to the recovery of the sparsest linear model in compressed sensing. 

### The Asymmetric Maximum Margin Bias of Quasi-Homogeneous Neural Networks

<https://www.arxiv.org/abs/2210.03820>

Define quasi-homogeneous models that is expressive enough to describe homogeneous activations, biases, residual connections, and normalization. Show that they also have marimum-margin bias, however the gradient flow favors a subset of parameters unlike homogeneous networks.

### Unifying and Boosting Gradient-Based Training-Free Neural Architecture Search

<https://www.arxiv.org/abs/2201.09785>

Show that the training-free NAS metrics are equal up to constant, which allows us to derive single generalzation bound results for all NAS metrics.

### Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent

<https://www.arxiv.org/abs/2207.04036>

Show that under natural condition of commuting parameterization, the reparameterized gradient flow model is equivalent to the continuous mirror descent with a related Legendere function.

### Learning threshold neurons via the "edge of stability"

<https://www.arxiv.org/abs/2212.07469>

Consider the simple model of two-layer NN with single neuron, trained with EoS learning rate. When trained on small learning rate, it fails to have 'threshold neuron' where EoS learning rate can. The threshold neuron is function that is zero near origin and abs(x) - b outside.

### Gaussian Pre-Activations in Neural Networks: Myth or Reality?

<https://www.arxiv.org/abs/2205.12379>

Derive the family of pairs of activation function and initialization distribution that ensure the pre-activation remain Gaussian throughout the network's depth, even in narrow neural networks. Discover a set of constraints that ensure this property, and build an exact Edge of Chaos analysis.

### A Dynamics Theory of Implicit Regularization In Deep Low-Rank Matrix Factorization

<https://www.arxiv.org/abs/2212.14150>

Introduce the landscape analysis to explain implicit regularization in deep low-rank matrix factorization, focusing on gradient region like saddle point and local minima. Theoretically establish the connection between saddle point escaping stages and the matrix rank, prove that DMF will converge to a second-order critical point after R stages of SPE where R is rank.

### On the Symmetries of Deep Learning Models and their Internal Representation

<https://www.arxiv.org/abs/2205.14258>

Find the symmetry group of DL model, which is the set of matrices A satisfying phi(Ax) = B phi(x), for activation phi, named interwiner group. 

### SGD and Weight Decay Provably Induces a Low-Rank Bias in Neural Networks

<https://www.arxiv.org/abs/2206.05794>

Show that the SGD with small batch size, high learning rate, and large weight decay induces the bias toward rank minimization over weight matrices.

### On the Global Convergence of Fitted Q-Iteration with Two-layer Neural Network Parametrization

<https://www.arxiv.org/abs/2211.07675>

Study the sample complex guarantee of fitted Q-iteration with two-layer NN parameterization. This estimates Q-function in each iteration with a convex optimization problem, and achieve order-optimal sample complexity, without any assumption on the MDP like linearity or low rank.

### Revisiting Estimation Bias in Policy Gradients for Deep Reinforcement Learning

<https://www.arxiv.org/abs/2301.08442>

Extend the study of estimation bias in the tabular and softmax case to DRL setting, and how this bias can lead to suboptimal policies. Show that the estimation bias can be reduced with small learning rate, adaptive optimizer, and KL regularization, and the first two methods makes the policy optimization robust ot the bias.

### Inductive Bias of Gradient Descent for Weight Normalized Smooth Homogeneous Neural Nets

<https://www.arxiv.org/abs/2010.12909>

Consider the weight normalized smooth homogeneous neural nets, especially standard weight normalization and exponential weight normalization. Show that EWN is equivalent to gradient flow on standard network with an adaptive learning rate, and establish asymptotic relation between weights and gradients for both SWN and EWN. Show that EWN perfers asymptotic relative sparsity, and give finite-time convergence rate of the loss.

### Simplicity Bias in 1-Hidden Layer Neural Networks

<https://www.arxiv.org/abs/2302.00457>

Show that the shallow neural network have bias towards depending on low dimensional projection of the inputs, by proving that if the data is linearly separable, the netwrok primarily depends on linearly separable subspace.

### Stochastic Gradient Descent-induced drift of representation in a two-layer neural network

<https://www.arxiv.org/abs/2302.02563>

Study the drift induced by the noise of SGD in a continual learning. By decomposing the learning dynamics into the normal and tangent space of the minimum loss manifold, show that former correspond to a finite variance fluctuation and the latter correspondence as an effective diffusion process. Analytically compute the fluctuation and the diffusion coefficient.

### How degenerate is the parametrization of neural networks with the ReLU activation function?

<https://www.arxiv.org/abs/1905.09803>

Show that the inverse stability, which allow to show that realization's local minimum property transfer to parameter's local minimum property fails in neural network, and present pathologies that prevents inverse stability. Propose restricted set of parameters for shallow neural network that is inverse stable w.r.t. Sobolev norm.

### The Geometry of Neural Nets' Parameter Spaces Under Reparametrization

<https://www.arxiv.org/abs/2302.07384>

Study the invariance quantities of neural nets under reparametrization of parameters, through Reimannian geomtry. Show that this invariance is an inherent property of any neural net.

### Generalization Bounds for Adversarial Contrastive Learning

<https://www.arxiv.org/abs/2302.10633>

Leverage the Rademacher complexity to analyze generalization performance of ACL, for the multi-layer neural network under Lp attack, which shows that the average adversarial risk of the downstream task can be upper bounded by the adversarial unsupervised risk.

### Learning to Generalize Provably in Learning to Optimize

<https://www.arxiv.org/abs/2302.11085>

Show that by incorporating local entropy and Hessian to learned optimizer framework, the generalization ability can be learned during meta-training and transformed to optimizee loss functions.

### Regression as Classification: Influence of Task Formulation on Neural Network Features

<https://www.arxiv.org/abs/2211.05641>

For two-layer ReLU networks, show that the regression formulation yields a measure whose support can differ greatly from that for classification. 

### Training neural networks with structured noise imporves classification and generalization

<https://www.arxiv.org/abs/2302.13417>

Show that adding structure into noisy training data can substantially improve memory performance, allowing perfect classification and maximal basins of attraction. Also prove that the unlearning rule coincides with the training-with-noise when noise is maximal.

### Provable Convergence of Tensor Decomposition-Based Neural Network Training

<https://www.arxiv.org/abs/2303.06815>

Show that the tensor decomposition of model compression can be solved by tensor block coordinate descent which is gradient free, and it globally converge the critical point.

### Phase Diagram of Initial Condensation of Two-layer Neural Networks

<https://www.arxiv.org/abs/2303.06561>

Present a phase diagram of condensation, which is a phenomenon where the weight vectors concentrate on isolated orientations during the training. This phase diagram serves to provide understanding of the dynamical regimes of NNs and their dependence.

### General Loss Functions Lead to (Approximate) Interpolation in High Dimensions

<https://www.arxiv.org/abs/2303.07475>

Show that the implicit bias of general family of convex losses in overparameterized regime, is approximated minimum-norm interpolation, similar to the squared loss.

### The Benefits of Mixup for Feature Learning

<https://www.arxiv.org/abs/2303.08433>

In the feature-noise model, show that Mixup training can effectively learn rare feature from its mixture, where the standard learning can only learn the common features. Also shows that mixup's benefit comes from early training phase.

### Theoretical guarantees for neural control variates in MCMC

<https://www.arxiv.org/abs/2304.01111>

Derive the optimal convergence rate of neural control variate against the Markov chain, under ergodicity assumption.

### Decomposing neural networks as mappings of correlation functions

<https://www.arxiv.org/abs/2202.04925>

Characterize the mapping between probability distributions by deep neural networks, as an iterated transformation of distributions. This allows to identify essential statistics, and show that correlations capture the information processing.

### Points of non-linearity of functions generated by random neural networks

<https://www.arxiv.org/abs/2304.09837>

Compute the expected distribution of the points of nonlinearity, which explains why the network be biased towards simpler geometry.

### Leveraging the two timescale regime to demonstrate convergence of neural neworks

<https://www.arxiv.org/abs/2304.09576>

For the case where the stepsize of inner layer is much smaller than outer layer, prove the convergence of the gradient flow to a global optimum, which requires much smaller number of neurons.

### Criticality versus uniformity in deep neural networks

<https://www.arxiv.org/abs/2304.04784>

In the edge of choas where training has maximum trainable depth, the tanh activation has uniformity in phase space, that the post-activation has maximum entropy, indicating that only edge of chaos is not sufficient.

### The Disharmony Between BN and ReLU Causes Gradient Explosion, but is Offset by the Correlation Between Activations

<https://www.arxiv.org/abs/2304.11692>

Show that ReLU reduces variance more than expected, and batch normalization amplifies the gradient during recovery of this. This makes gradient explosion even if the forward computation is stable. 

### Backward Feature Correction: How Deep Learning Performs Deep (Hierarchical) Learning

<https://www.arxiv.org/abs/200.04413>

Establish backward feature correction where lower-level error is corrected when trained together with higher layers, and show that this principle is necessary by finding a concept class of multi-variate polynomials that neural netwroks can learn in polytime via SGD, while other models like layerwise training, composition of kernels, or two-layer networks can't learn the solution.

### How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model

<https://www.arxiv.org/abs/2307.02129>

Consider random hierarchy model, where each C classes correspond to m synonymic composition of high-level featueres that are composed of sub-features through an iterative process repeated L times, show that C * m^L samples are needed for deep CNN to learn. This value coincide the training set size that representation becomes invariant to exchanges of synonyms, also the number of data where correlations between low-level features and classes are detecable. This indicate that deep CNNs overcome the curse of dimensionality by building invariant representations.

### Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD

<https://www.arxiv.org/abs/2307.00310>

Develop a new analysis for DP-SGD that captures the intuition that similar neighbors in dataset enjoy better privacy than outliers, by introducing the dependence on the distribution of model. Develop new composition theorem to use this per-step analysis, shows that this analysis shows DP-SGD leaks significanly less privacy for many datapoints, and correctly classified points obtain better privacy.

### Neural Characteristic Activation Value Analysis for Improved ReLU Network Feature Learning

<https://www.arxiv.org/abs/2305.15912>

Consider the characteric activation which is nondifferentiable points induced by ReLU, and find a connection between these characteristic set and learned features. Use this to understand the effects of normalization on regularization and stabilization of SGD.

### The Implicit Bias of Minima Stability in Multivariate Shallow ReLU Networks

<https://www.arxiv.org/abs/2306.17499>

Extend the univariate implicit bias that linearly stable minima correspond to bounded weighted L1 norm of second derivative, which gets smaller for larger step sizes, to the multivariate case which applies to the Laplacian. Also show that the universal approximation do not hold for stable shallow networks, while two hidden-layer can. Finally prove that universal approximation holds for sufficiently smooth functions to the stable shallow ReLU networks.

### Neural networks can detect model-free static arbitrage strategies

<https://www.arxiv.org/abs/2306.16422>

Show that neural network can detect model-free static arbitrage strategy if exists. Prove that a single neural network can approximately solve a class of convex semi-infinite programs, which shows the main result.

### Any Deep ReLU Network is Shallow

<https://www.arxiv.org/abs/2306.11827>

Constructively prove that any deep ReLU networks can be rewritten with three-layer ReLU net with weights valued in the extended reals.

### The Implicit Bias of Batch Normalization in Linear Models and Two-layer Linear Convolutional Neural Networks

<https://www.arxiv.org/abs/2306.11680>

Show that linear model with batch normalization converges to a uniform margin classifier, and two-layer single filter linear CNN also converges towards patch-wise uniform margin. 

### Understanding Generalization in the Interpolation Regime using the Rate Function

<https://www.arxiv.org/abs/2306.10947>

Propose real-valued function formulation of smoothness, and use this to show that various learning techniques like SGD, L2 regularization, augmentation, invariant architecture, and overparameterization bias toward smooth interpolators and with better generalization error.

### Training shallow ReLU networks on noisy data using hinge loss: when do we overfit and is it benign?

<https://www.arxiv.org/abs/2306.09955>

Under the two-layer ReLU net trained with GD on hinge loss with corrupted or flipped small portion, find that there is two phase of training dynamics that first phase fits clean points close to zero, and second phase either fits the corrupt points or zero-ed.

### Exact Count of Boundary Pieces of ReLU Classifiers: Towards the Proper Complexity Measure for Classification

<https://www.arxiv.org/abs/2306.08805>

Measure the complexity of decision boundary ith number of affine pieces, and develop the method to count the exact number of boundary pieces. 

### Provable Identifiability of Two-Layer ReLU Neural Networks via LASSO Regularization

<https://www.arxiv.org/abs/2305.04267>

Show that under LASSO regularization, two-layer ReLU network can learn the small subset of feature that determines output when number of sample scales logarithmically. 

### Mechanism of feature learning in deep fully connected networks and kernel machines that recursively learn features

<https://www.arxiv.org/abs/2212.13881>

Identify the feature learning of neural network that implements average gradient outer product to up-weight features strongly related to model output. 

### Phase transitions in the mini-batch size for sparse and dense neural networks

<https://www.arxiv.org/abs/2305.06435>

Show that there exists critical value of batch size in teacher student scenario of two layer network, that under this value the network fails to be trained, while higher batch size gives perfect learning or good generalization.

### Deep ReLU Networks Have Surprisingly Simple Polytopes

<https://www.arxiv.org/abs/2305.09145>

Empirically find that ReLU networks has simple polytopes at initialization and during gradient descent. And derive combinatorial computation to explain why deeper networks do not create more complicated polytopes, by bounding the average number of faces of polytopes with the function of dimension.

### Complexity of Feed-Forward Neural Networks from the Perspective of Functional Equivalence

<https://www.arxiv.org/abs/2305.11417>

Using the permutation invariance property, derive novel covering number bounds for FFNNs, that reduces the complexity of neural network. 

### Neural networks trained with SGD learn distributions of increasing complexity

<https://www.arxiv.org/abs/2211.11567>

Show that SGD training for NNs initially classify their inuts with lower-order statistics like mean and variances, and expoit higher-order statistics later. 

### Learning Two-Layer Neural Networks, One (Giant) Step at a time

<https://www.arxiv.org/abs/2305.18270>

For single direction learning, show that batch size d is enough for feature learning, while multiple direction needs d^2, and hard directions that has no first l Hermite coefficients, the batch size h^l is needed. After some steps, d batch size is enough to learn new targt directions spanning the subspace.

### Abide by the law and follow the flow: conservation laws for gradient flows

<https://arxiv.org/abs/2307.00144>

Derive an algorithm that finds the conserved quantities of gradient flow when training neural networks, and also an algorithm that upper bounds the number of conserved quantities. Show that known conserved quantities for linear net and ReLU nets are complete.

### Finite-Time Analysis of Whittle Index based Q-Learning for Restless Multi-Armed Bandits with Neural Network Function Approximation

<https://arxiv.org/abs/2310.02147>

Consider the finite-time analysis of neural Whittle index policy which uses neural network to approximate the Whittle index, show that the convergence rate is k(-2/3) where k is number of iteration.

### Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems

<https://arxiv.org/abs/2303.12981>

Show that the superlevel set of policy optimization is connected in both tabular setting and neural policies, and also show that reward satisfies equiconnectedness property. Use this to derive minimax theorem for robust RL.

### A Theory of Unsupervised Translation Motivated by Understanding Animal Communication

<https://arxiv.org/abs/2211.11081>

Consider the theory of understanding unsupervised translation without parallel translation, or even the related subject domain or similar linguistic structure. Proves bound on necessary sample complexity in two stylized models, show that the error rate inversely related on langauge complexity and the number of common ground.

### Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks

<https://arxiv.org/abs/2305.06986>

Consider the hierarchical target function, derive the sample complexity's upper bound and width to achieve low test error, and show that three-layer network under feature-learning regime has improved sample complexity compared to two-layer network, by learning the non-linear features. Also derive the optimisation based depth separation via a target function that is efficiently learnable on three-layer network but not on two-layer network.

### Transformers learn through gradual rank increase

<https://arxiv.org/abs/2306.07042>

Prove that the difference between trained ans initial weights progressively increases in rank for weights in transformers, under diagonal weight matrices and small initialisation.

### Implicit Bias of Gradient Descent for Logistic Regression at the Edge of Stability

<https://arxiv.org/abs/2305.11788>

Show that the gradient descent for logistic regression converges even in the EoS regime despite the oscillation, and the projected solution converges to a solution that minimizes some strongly convex potential. Also show that similar setting can diverge under the exponential loss.

### Provable Guarantees for Neural Networks via Gradient Feature Learning

<https://arxiv.org/abs/2310.12408>

Derive analysis for gradient based learning, derives the performance guarantee of two layer network. Apply this analysis to Gaussian mixture, parity, function, and show that the feature learning outperforms the kernel regime.

### Don’t blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy

<https://arxiv.org/abs/2308.12553>

Show that the default ERMs still uses shortcut learning in the case of linear perception, because their implicit bias towards maximum margin leads to shortcut than the stable feature. Instead, shows that uniform margin inductive bias is more robust to shortcut learning.

### Bottleneck Structure in Learned Features: Low-Dimension vs Regularity Tradeoff

<https://arxiv.org/abs/2305.19008>

Show that the DNNs with large depth, and with regularisation, minimises notion of rank of learned function, which bounds the pseudo-determinant of Jacobian and subadditive under composition and addition. This gives balance between low dimensional representation and complexity in feature maps. Then prove this bottlenecked structure in infinite depth, so that almost all features are low dimensional. 

### Training shallow ReLU networks on noisy data using hinge loss: when do we overfit and is it benign?

<https://arxiv.org/abs/2306.09955>

Consider the linearly separable data with small portion of flipped labels, show that the two-layer ReLU network trained with GD on hinge loss has two phases of training dynamics, where the first phase clean points are fitted with training loss close to zero, then the second phase the clean points oscillates around boundary of zero loss, and corrupt points converge to zero loss or are zeroed.

### A Theory of Multimodal Learning

<https://arxiv.org/abs/2309.12458>

Propose a theoretical framework for multimodal learning problem, where they show superior generalisation bound compared to unimodal setting up to O(sqrt(n)), the sample size. This happens when there is connection and heterogenity between the modalities.

### Phase diagram of early training dynamics in deep neural networks: effect of the learning rate, depth, and width

<https://arxiv.org/abs/2302.12250>

Analyses the maximum eigenvalue of Hessian in training dynamics, find that it has four regimes, where first two regimes show phase diagram depending on depth, width, and maximum eigenvalue at initialisation.

### Provable Advantage of Curriculum Learning on Parity Targets with Mixed Inputs

<https://arxiv.org/abs/2306.16921>

Assuming that the data distribution is mixture of sparse and dense inputs, show that curriculum learning with sparse inputs first can optimize 2-layer ReLU NNs to learn parities of large degrees, while without curriculum, any fully connected NNs cannot learn.

### Convergence of Actor-Critic with Multi-Layer Neural Networks

<https://openreview.net/forum?id=QlfGOVD5PO>

Show that the arbitrarily deep neural network that are projected initially, converges to the neighbors of the policy with gradient being small enough, when trained with actor-critic algorithm.

### Simplicity Bias in 1-Hidden Layer Neural Networks

<https://arxiv.org/abs/2302.00457>

Show that the shallow neural networks shows simplicity bias, that it depends on only low dimensional projection, thereby converging to possibly non-robust solution.

### On the spectral bias of two-layer linear networks

<https://openreview.net/forum?id=FFdrXkm3Cz>

Consider two-layer overparameterised linear network trained with gradient flow, with first weights initialised to be orthogonal, shows that the solution converges to minimum l2-norm solution, and the first layer's singular values are almost zero for small init variance, while being isotropic for large init variance.