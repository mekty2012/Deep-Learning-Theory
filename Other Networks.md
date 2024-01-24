## Sequence Models

### On the stability properties of Gated Recurrent Units neural networks

<https://www.arxiv.org/abs/2011.06806>

Provide sufficient conditions for guaranteeing the Input-to-State Stability and the Incremental Input-to-State Stability of GRUs. 

### Theory of gating in recurrent neural network

<https://www.arxiv.org/abs/2007.14823>

Show that gating offers flexible control of two salient features, timescales and dimensionality.

### MomentumRNN: Integrating Momentum into Recurrent Neural Networks

<https://www.arxiv.org/abs/2006.06919>

Establish a connection between the hidden state dynamics in an RNN and gradient descent, integrating the momentum to this framework, prove that MomentumRNNs alleviate the vanishing gradient issue.

### Transformer Vs. MLP-Mixer Exponential Expressive Gap For NLP Problems

<https://www.arxiv.org/abs/2208.08191>

Analyze the expressive power of mlp-based architectures in modelling dependencies between multiple different inputs, and show an exponential gap between the attention and the mlp-based mechanisms.

### Universality and approximation bounds for echo state networks with random weights

<https://www.arxiv.org/abs/2206.05669>

For echo state network with only its readout weights are optimized, show that they are universal under weak conditions for the continuous casual time-invariant operators.

### Your Transformer May Not be as Powerful as You Expect

<https://www.arxiv.org/abs/2205.13401>

Analyze the power of relative positional encoding based Transformer, show that there exists continuous seq2seq function that RPE-transformer cannot approximate no matter how deep and wide is. The key reason is that most RPEs are placed in the softmax attention so generate a right stochastic matrix, which restricts the network from capturing positional information. Find sufficient condition for universal approximation of Transformer, and suggest attention module satisfying this.

### Memory Capacity of Recurrent Neural Networks with Matrix Representation

<https://www.arxiv.org/abs/2104.07454>

Define a probabilistic notion of memory capacity based on Fisher information for RNNs, and show that this memory capacity is usually bounded by size of state matrix. Show that the models with external state memory has increase in memory capacity.

### Recurrent Neural Networks and Universal Approximations of Bayesian Filters

<https://www.arxiv.org/abs/2211.00335>

Consider the Bayesian optimal filtering problem, that estimates conditional statistics of a latent time series signal, using the RNN. Provide the approximation error bounds that is time-uniform.

### A Theoretical Understanding of shallow Vision Transformers: Learning, Generalization, and Sample Complexity

<https://www.arxiv.org/abs/2302.06015>

Prove the sample complexity of shallow vision transformer, which is correlated with the inverse of fraction of label-relevant tokens, token noise level, and initial model error. Also show that SGD leads sparse attention map.

### Transformers Learn Shortcuts to Automata

<https://www.arxiv.org/abs/2210.10749>

Show that transformer can represent any finite state automaton by hierarchically reparameterizing the dynamics, and there is shortcut solution with o(T) layer where T is input sequence length, and there always exist polynomial-siazed O(log T) depth solution. Moreover O(1) simulators are common.

### One Step of Gradient Descent is Provably the Optimal In-Context Learner with One Layer of Linear Self-Attention

<https://www.arxiv.org/abs/2307.03576>

Show that under the standard Gaussian covariates, the single layer of linear self-attention trained with noisy linear regression data will implement a single step of GD on the least-squares linear regression. Then shows changes in the distribution changes learned algorithm a lot, the pre-conditioned GD while response has not large effects.

### Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection

<https://www.arxiv.org/abs/2306.04637>

Show that transformers can implement least squares, ridge regression, Lasso, generalized linear models and two-layer neural network trained with gradient descent. Then show that transformers can implement complex ICL procedures that selects algorithm in-context.

### Why can neural language models solve next-word prediction? A mathematical perspective

<https://www.arxiv.org/abs/2306.17184>

Study a class of formal languages that model the real-world examples of English sentences, and construct neural language models can solve the next-word prediction task in this setting with zero error. 

### Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective

<https://www.arxiv.org/abs/2305.15408>

Show that bounde depth transformer are unable to produce correct answers for basic arithmetics unless model grows polynomially with input ength, where constant size transformer can generate CoT derivation with commonly used language format. Moreover, show that dynamic programming can be solved with CoT.

### Trained Trnasformers Learn Linear Models In-Context

<https://www.arxiv.org/abs/2306.09927>

Investigate the dynamics of ICL in transformers with single linear self-attention layer on linear regression, and show that the gradient flow finds a global minimum which is best linear predictor.

### On the Convergence of Encoder-only Shallow Transformers

<https://arxiv.org/abs/2311.01575>

Show that encoder-only shallow transformer converges globally, in finite width regime, handling the softmax operation, which requires only quadratic overparameterisation. 

### Representational Strengths and Limitations of Transformers

<https://arxiv.org/abs/2306.02896>

Show that in the sparse averaging task, both RNN and FCNN requires polynomially scaling complexity, where transformer requires logarithmic scale in the input size, and show the necessity of large embedding dimension. Also show the negative result where the triple detection task requires linearly scaling complexity.

### Transformers are uninterpretable with myopic methods: a case study with bounded Dyck grammars

<https://arxiv.org/abs/2312.01429>

Consider synthetic task named as bounded Dyck grammar, and show that while transformer can solve this tasks, because the geometry of global optimum is large, there is another parameterisation that attention pattern in a layer can be randomised, while preserving the function, showing that the usual interpretation based on attention can be misleading.

## PINN

### When Do Extended Physics-Informed Neural Networks (XPINNs) Improve Generalization?

<https://www.arxiv.org/abs/2109.09444>

Provide a prior generalization bound via the complexity of the target functions in the PDE problem, and a posterior generalization bound. Show that domain decomposition which decompose solution to simpler parts and make easier to solve, introduces a tradeoff for generalization, where the decomposition leads to less training data being available in each subdomain, prone to overfitting.

### Self-scalable Tanh (Stan): Faster Convergence and Better Generalization in Physics-informed Neural Networks

<https://www.arxiv.org/abs/2204.12589>

Propose self-scalable Tanh activation for PINNs, show that PINNs with Stan have no spurious stationary points when using gradient descent algorithms.

### Certified machine learning: A posteriori error estimation for physics-informed neural networks

<https://www.arxiv.org/abs/2203.17055>

Assuming that the underlying differential equation is ODE, derive a rigorous upper limit on the PINN prediction error, for arbitrary point without knowing the solution.

### Is L2 Physics-Informed Loss Always Suitable for Training Physics-Informed Neural Networks?

<https://www.arxiv.org/abs/2206.02016>

Show that for Hamilton-Jacobi-Bellman equations, for general Lp Physics informed loss, the equation is stable only if p is sufficiently large. Hence it is better to choose L-infinity loss.

### Estimates on the generalization error of Physics Informed Neural Networks (PINNs) for approximating PDEs

<https://www.arxiv.org/abs/2006.16144>

Provide upper bound on the generalization error of PINNs approximating solutions of the forward problem for PDEs.

### Convergence analysis of unsupervised Legendre-Galerkin neural networks for linear second-order elliptic PDEs

<https://www.arxiv.org/abs/2211.08900>

ULGNet express solution as a spectral expansion w.r.t. Legendre basis and predict the coefficient with DNNs. Prove that the minimizer of discrete loss function converges to the weak solution of the PDEs.

## Error estimates for physics informed neural networks approximating the Navier-Stokes equations

<https://www.arxiv.org/abs/2203.09346>

Prove the rigorous bounds on the errors resulting from the approximation of the incompressible Navier-Stokes equations with (X)PINNs. Show that the PDE residual can be arbitrarily small for tanh neural networks with two hidden layers.

### Error Analysis of Physics-Informed Neural Networks for Approximating Dynamic PDEs of Second Order in Time

<https://www.arxiv.org/abs/2303.12245>

Consider the approximation of PDE of second order in time, by PINN and provide an error analysis of PINN for the wave equation, Sine-Gordon equation and the linear elastodynamic equation. Show that the shallow Tanh-network's error for the solution field, time derivative and gradient field can be effectively bounded by the training loss and number of data points. Also suggest new form of training loss, which contain residuals that are crucial to the error estimate.

### On the Compatibility between Neural Networks and Partial Differential Equations for Physics-informed Learning

<https://www.arxiv.org/abs/2212.00270>

Prove that a linear PDE up to n-th order can be satisfied by an MLP with Cn activation function when the weights lie on a certain hyperplane. Such hyperplane equipped MLP becomes physics-enforced, which no longer requires the loss function for the PDE itself.

### Transferability of Winning Lottery Tickets in Neural Network Differential Equation Solvers

<https://www.arxiv.org/abs/2306.09863>

Extend the lottery ticket hypothesis and elastic lottery hypothesis to Hamiltonian NN for solving differential equations, find lottery tickets for two Hamiltonian Neural Networks with transferability and dependent accuracy on integration times.

### Global Convergence of Deep Galerkin and PINNs Methods for Solving Partial Differential Equations

<https://www.arxiv.org/abs/2305.06000>

Show that as the width of network goes to infinity, the trained neural network converges to the solution of infinite-dimensional ODE, and the PDE residual converges to zero. 

### Provably Correct Physics-Informed Neural Networks

<https://www.arxiv.org/abs/2305.10157>

Establish tolerance based correctness condition instead of comparison to ground truth points, and design partial CROWN that ost-train to bound PINN residual errors.

## Invertible NN

### Understanding and Mitigating Exploding Inverses in Invertible Neural Networks

<https://www.arxiv.org/abs/2006.09347>

Show that commonly used INN architectures suffer from explodinig inverses, and reveal failures including the non-applicability of the change-of-variables formula on in- and OOD data, incorrect gradients, inability to sample from normalizing flow. 

## Lipschitz-Net

### Pay attention to your loss: understanding misconceptions about 1-Lipscitz neural networks

<https://www.arxiv.org/abs/2104.05097>

Show that the 1-Lipscitz networks are as accuracte as classical one, and can fit arbitrarily difficult boundaries. Then show these 1-Lipscitz neural networks generalize well under milder assumptions, and finally show that hyper-parameters of the loss are crucial for controlling the accuracy-robustness trade-off.

### Rethinking Lipschitz Neural Networks for Certified L-infinity Robustness

<https://www.arxiv.org/abs/2210.01787>

Show that using the norm-bounded affine layers and Lipschitz activation lose the expressive power, while other Lipschitz networks like GroupSort and L-infinity networks bypass these impossibilities.

### Improving Lipschitz-Constrained Neural Networks by Learning Activation Functions

<https://www.arxiv.org/abs/2210.16222>

Study the networks with learnable 1-Lipschitz linear spline activations, and show that they are the solutions of a functional optimization problem with second-order total-variation regularization.

### A Unified Algebraic Perspective on Lipschitz Neural Networks

<https://www.arxiv.org/abs/2303.03169>

Introduce a novel algebraic perspective that unifies various types of 1-Lipschitz neural networks, which shows that many existing techniques can be derived and generalized via finding analytic solutiosn of a semidefinite programming.

## Implicit Models

### Fixed points of nonnegative neural networks

<https://www.arxiv.org/abs/2106.16239>

Derive condition for the existence of fixed points of nonnegative neural networks, by recognizing them as monotonic scalable functions within nonlinear Perron Frobenius theory, and show fixed point set's shape is often interval.

### Fixed Points of Cone Mapping with the Application to Neural Networks

<https://www.arxiv.org/abs/2207.09947>

The cone mappings are often modelled with non-negative weight neural networks, however the nonnegative data usually do not guarantee nonnegative weight, hence this assumption often fails, and require weakening on the assumption for fixed point, scalability. Derive condition for the existence of fixed points of cone mappings without assuming scalability of functions, therefore available to applied to such NNs.

### A global convergence theory for deep ReLU implicit networks via over-parameterization

<https://www.arxiv.org/abs/2110.05645>

Show that a randomly initialized gradient descent converges to a global minimum at a linear rate for the square loss function if the implicit neural network is over parameterized.

### On the optimization and generalization of overparameterized implicit neural networks

<https://www.arxiv.org/abs/2209.15562>

Usual analysis on implicit neural network collapse to studying only last layer, so study the case when optimizing the implicit layer only. Show that global convergence is guaranteed, and give generalization that is initialization sensitive.

### Global Convergence Rate of Deep Equilibrium Models with General Activations

<https://www.arxiv.org/abs/2302.05797>

Show that the DEQs with any general activation having bounded first and second derivative globally converges if it is over-parameterized, by designing the population Gram matrix and dual activation by Hermite polynomial. 

### PAC bounds of continuous Linear Parameter-Varying systems related to neural ODEs

<https://www.arxiv.org/abs/2307.03630>

Show that large class of neural ODEs can e embedded into linear parameter varying systems, and derive PAC bounds under stability, which does not depend on the integration interval.

## Others

### Deep Autoencoders: From Understanding to Generalization Guarantees

<https://www.arxiv.org/abs/2009.09525>

Reformulate AEs by continuous piecewise affine structure, to show how AEs approximate the data manifold, giving some insights for reconstruction guarantees and interpretation of regularization guarantees. Design two new regularization that leverages the inherent symmetry learning, prove that the regularizations ensure the generalization with assumption on symmetry of the data with Lie group.

### The dynamics of representation learning in shallow, non-linear autoencoders

<https://www.arxiv.org/abs/2201.02115>

Derive a set of asymptotically exact equations that describe the generalisation dynamics of autoenoders trained with SGD in the limit of high-dimensional inputs. 

### De Rham compatible Deep Neural Networks

<https://www.arxiv.org/abs/2201.05395>

Construct classes of neural networks with ReLU and BiSU activation, emulating the lowest order Finite Element spaces on regular simplicical partitions of polygonal domains for 2, 3 dimension. 

### Optimal training of integer-valued neural networks with mixed integer programming

<https://www.arxiv.org/abs/2009.03825>

Formulate new MIP model improving the training efficiency which can train the integer-valued neural networks, with optimization of the number of neurons and batch training.

### Concentration inequalities and optimal number of layers for stochastic deep neural networks

<https://www.arxiv.org/abs/2206.11241>

State the concentration and Markov inequality for output of hidden layers and output of SDNN. This introduce expected classifier, and the probabilistic upper bound for the classification error. Also state the optimal number of layers by optimal stopping procedure.

### Diversity and Generalization in Neural Network Ensembles

<https://www.arxiv.org/abs/2110.13786>

Provide sound answers to the following questions, how to measure diversity, how diversity relates to the generalization error of an ensemble, and how diversity is promoted by neural network ensemble algorithms.

### On a Sparse Shortcut Topology of Artificial Neural Networks

<https://www.arxiv.org/abs/1811.09003>

Propose new shortcut architecture, and show that it can approximate any univariate continuous function in width-bounded setting, and show the generalization bound.

### A Kernel-Expanded Stochastic Neural Network

<https://www.arxiv.org/abs/2201.05319>

Design new architecture which incorporates support vector regression at its first layer, allowing to break the high-dimensional nonconvex training of neural network to series of low-dimensional convex optimization, and can be trained using imputation-regularized optimization, with a theoretical guarantee to global convergence.

### Deep Layer-wise Networks Have Closed-Form Weights

<https://www.arxiv.org/abs/2202.01210>

Show that layer-wise network, which trains one layer at a time, has a closed form weight given by kernel mean embedding with global optimum. 

### A Closer Look at Learned Optimization: Stability, Robustness, and Inductive Biases

<https://www.arxiv.org/abs/2209.11208>

Use tools from dynamical systems to analyze the inductive bias and stability of the optimization algorithms, which allows us to design inductive biases for blackbox optimizers. Then apply this to noisy quadratic model and introduce modification on learned optimizer.

### A theory of learning with constrained weight-distribution

<https://www.arxiv.org/abs/2206.08933>

Derived an analytic solution of the memorical capacity of perceptron with constraints on its weights, which shows that the reduction is related to the Wasserstein distance between the imposed distribution and the standard normal distribution.

### KAM Theory Meets Statistical Learning Theory: Hamiltonian Neural Networks with Non-Zero Training Loss

<https://www.arxiv.org/abs/2102.11923>

Hamiltonian neural network, which approximates the Hamiltonian with neural network, is perturbation from the true dynamics under non-zero loss. To apply perturbation theory for this, called KAM theory, provide a generalization error bound for Hamiltonian neural networks by deriving an estimate of the covering number of the gradient of the MLP, then giving L infinity bound on the Hamiltonian.

### Bounding The Rademacher Complexity of Fourier Neural Operator

<https://www.arxiv.org/abs/2209.05150>

Investigate the bounding of Rademacher complexity of FNO based on some group norms, and the generalization error of the FNO models. 

### A Convergence Rate for Manifold Neural Networks

<https://www.arxiv.org/abs/2212.12606>

Show that the manifold NN, which uses spectral decomposition of Laplace Beltrami operator to simulate neural network in non-Euclidean domain, converges to the continuum limit as the number of sample points tends to infinity. This rate of convergence depends on the intrinsic dimension, but is independent of the ambient dimension, and the rate dependency of depth and filter number is discussed.

### Fundamenatl Limits of Two-Layer Autoencoders, and Achieving Them with Gradient Methods

<https://www.arxiv.org/abs/2212.13468>

Focus on the two-layer Nonlinear Autoencoder, whose input dimension scales linearly with the size of the representation, and show the structure of minimizers and that the minimizers can be achieved with gradient methods. For the sign activation function, show the fundamental limit of lossy compression of Gaussian sources.

### Neural Network Architecture Beyond Width and Depth

<https://www.arxiv.org/abs/2205.09459>

Design the network named as NestNet, which uses the neural network as the network. Show that height s NestNet require n parameters to approximate 1-Lipschitz continuous function with error O(n^(-(s+1)/d)), where the standard ReLUNet has O(n^(-2/d)).

### Lower Bounds on the Depth of Integral ReLU Neural Networks via Lattice Polytopes

<https://www.arxiv.org/abs/2302.12553>

Prove that the integer weight ReLU nets have strictly increasing functions with the network depth for arbitrary width. Show that log(n) layers are required to compute maximum of n numbers, which matches known upper bound. Uses duality between neural networks and Newton polytopes via tropical geometry.

### Globally Optimal Training of Neural Networks with Threshold Activation Functions

<https://www.arxiv.org/abs/2303.03382>

Study the weight decay regularized training can be formulated as a convex optimization.

### A Theoretical Understanding of Shallow Vision Transformers: Learning, Generalization, and Sample Complexity

<https://www.arxiv.org/abs/2302.06015>

Consider shallow ViT which one self-attention and two-layer perceptron. Characterize the sample complexity for zero generalization error which depends on inverse of fraction of label-relevant tokens, token noise level, and the initial model error. Also show that SGD leads to sparse attention map, and indicates proper token sparsification improve the test performance.

### Generalization and Estimation Error Bounds for Model-based Neural Networks

<https://www.arxiv.org/abs/2304.09802>

Leverage the complexity measures using the global and local Rademacher complexity, provide the upper bound on the generalization erros of model-based networks such as sparse coding and compressed sensing. Show that generalization abilities of these models outperform the regular ReLU networks, and derive design rule that allow to construct model-based networks with higher generalization.

### Differentiable Neural Networks with RePU Activtion: with Applications to Score Estimation and Isotonic Regression

<https://www.arxiv.org/abs/2305.00608>

Consider the rectified power unit neural network, whose partial derivatives are mixed-activated RePU neural networks, and derive the upper bound of the complexity of function class. Also establish the error bound for approximating C^s functions and derivatives, with the case when approximating in the low-dimensional support. 

### Generalization Bounds for Neural Belief Propagation Decoders

<https://www.arxiv.org/abs/2305.10540>

Consider the neural belief propagation which unfolds belief propagation to neural network, and show the generalization gap depending on the decoder complexity, including blocklength, message length, variable node degrees, decoding iteration.

### Universal Approximation and the Topological Neural Network

<https://www.arxiv.org/abs/2305.16639>

Consider TNN whose input is over Tychonoff topological space, and distributional neural network which takes Borel measures, show that these neural networks can approximate unfiromly continuosu functions with unique uniformity.

### A Unified Framework for U-Net Design and Analysis

<https://arxiv.org/abs/2305.19638>

Analysse the U-net architecture, show the role of encoder and decoder, and their high-resolution scaling limits.

### Neural Oscillators are Universal

<https://arxiv.org/abs/2305.08753>

Show that the neural oscillator which is ODE of order 2, can approximate causal and continuous operator on compact domain, with enough depth.