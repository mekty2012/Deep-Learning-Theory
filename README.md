# Ideas

## Genetic Programming for MAP

There is a SA based MA inference algorithm in Anglican. If so, why not genetic algorithm?

<https://probprog.github.io/anglican/assets/pdf/tolpin-socs-2015.pdf>

The algorithm known as Bayesian Ascent Monte Carlo exactly catches this problem. I think Variational Bayesian Ascent Monte Carlo also works well.

## Towards a liquid type for probabilistic programming

We have various liquid type in traditional programming langauge. Main fallacy in probabilistic programming is that testing is really hard. How can we?

<https://arxiv.org/pdf/2010.07763.pdf>

This paper gives refinement type on the basis of separation logic. Then, can we do similar thing on probabilistic separation logic?

<https://plv.mpi-sws.org/refinedc/paper.pdf>

Maybe following paper contains these things.

<https://arxiv.org/abs/1711.09305>

## Software Testing on probabilistic programming

Software testing works very well for problems as liveness and safety. However, hyperproperties are not that simple, like probability or security.
Can we extend software testing, like importance software testing, so that we can safely test probability-related hyperproperties?

<https://www.cs.cornell.edu/fbs/publications/Hyperproperties.pdf>

### Computability in Probabilistic Programming Language

From the definition of Anglican, it is obvious that ProbProg has either density function or mass function. Can we extend this scope? If not, what random variables can be computed?

<https://danehuang.github.io/papers/compsem.pdf>

<https://arxiv.org/pdf/2101.00956.pdf>

## Extending Inference Compilation

So LSTM is quite classical model, and we have much stronger models like transformer. What about using them?

If model is very complex, it will be hard for model to learn correct inference. By turning off some observe statement, can we do curriculum learning?

## Deep Learning Ideas

1. Can Neural Network learn arbitrary rules? In specific, design some random decision tree on image dataset. Will neural network learn it?
2. Can Neural Network learn length mapping? i.e., every input-output sequence is random vector with arbitrary length, however their length is always equal.

## Stochastic Differential Equation

Can this be effectively solved with help of probabilistic programming?

The paper seems highly dependent to discretization. Adopting idea from Neural Operator, can we have mesh independent solution?

## Delta Debugging for LaTeX

If possible... hierarchical!

Is there grammar for LaTeX? 

## SVI but with weighted sum of KLdivergence

If p is target and q is model, 
KL(q || p) is mode seeking (let q to concentrate on mode)
where KL(p || q) is mode covering (let q to cover mode(and submodes))
Then if, we run SVI with 
KL(q || p) + l * KL(p || q)?

usually, SVI uses mode seeking, amortized inference uses mode covering.

### Netsorizer

NETSOR program allows us to compute the infinite width limit of coordinate in neural network, using PL-theoretic approach.
Based on this computation, we know that 
1. Any NETSOR program follows Gaussian Process distribution
2. Any NETSOR program's NTK converges to deterministic kernel
3. Any NETSOR program's NTK converges to deterministic, during training
4. Any NETSOR program's weight and diagonal matrix of vectors satisfy asymptotic freeness, which is non-abelian analog of independence.
5. If some NETSOR program is BP-like, it satisfies gradient independence assumption.

Our approach is to create some static analyzer on PyTorch, Tensorflow, etc... that checks

1. Whether this program can be expressed as NETSOR program
2. Moreover to check NTKtrain property, we need to check the loss. I'm not sure on this actually...
3. If possible, we can also check the feature training property. Since the formula was quite simple, I think we may apply this to universal architectures...
4. Finally, to ease the proof, we can support BP-likeness analyzer.

### CLT Review list

SGD Generalizes Better Than GD (And Regularization Doesn't Help)
Generalizing Complex Hypotheses on Product Distributions: Auctions, Prophet Inequalities, and Pandora's Problem
A Theory of Heuristic Learnability
A Statistical Taylor Theorem and Extrapolation of Truncated Densities
\*Provable Memorization via Deep Neural Networks using Sub-linear Parameters
\*Functions with average smoothness: structure, algorithms, and learning
\*Non-asymptotic approximations of neural networks by Gaussian processes


\*Learning a Single Neuron with Gradient Methods
\*Universal Approximation with Deep Narrow Networks
How to trap a gradient flow
\*Implicit Bias of Gradient Descent for Wide Two-layer Neural Networks Trained with the Logistic Loss
\*A Corrective View of Neural Networks: Representation, Memorization and Learning
Implicit regularization for deep neural networks driven by an Ornstein-Uhlenbeck like process
\*Learning Over-parametrized Two-layer ReLU Neural Networks beyond NTK

Is your function low dimensional?
The implicit bias of gradient descent on nonseparable data
How do infinite width bounded norm networks look in function space?




Robustness

Query complexity of least absolute deviation regression via robust uniform convergence
The Sample Complexity of Robust Covariance Testing
Random Graph Matching with Improved Noise Robustness
Adversarially Robust Learning with Unknown Perturbation Sets
Robust Online Convex Optimization in the Presence of Outliers
Robust learning under clean-label attack
Adversarially Robust Low Dimensional Representations
Tsallis-INF: Improved Robustness to Adversarial Corruptions in Stochastic Multi-armed Bandits and Beyond
Outlier-Robust Learning of Ising Models Under Dobrushin's Condition
A Law of Robustness for Two-Layers Neural Networks
Corruption-robust exploration in episodic reinforcement learning
