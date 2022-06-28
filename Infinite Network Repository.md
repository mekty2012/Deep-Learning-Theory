# NNGP Correspondence

### Deep neural networks as Gaussian processes

<https://www.arxiv.org/abs/1711.00165>

In the limit of infinite network width, the neural network becomes a Gaussian process.

### Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes

<https://www.arxiv.org/abs/1910.12478>

Compute convergence to coordinate distribution for simple network. Can prove NNGP correspondence for universal architectures.

### On Infinite-Width Hypernetworks

<https://www.arxiv.org/abs/2003.12193>

Studies GP, NTK behavior of hypernetwork, which computes weights for a primiary network.

### Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes

<https://www.arxiv.org/abs/1810.05148>

Compute CNN both with and without pooling layers' equivalence to GP. Also introduce Monte Carlo method to estimate the GP to corresponding NN architecture.
Without pooling, weight sharing has no effect in GP, iplying that translation equivariance has no role in bayesian infinite limit. 

### Gaussian Process Behavious in Wide Deep Neural Networks

<https://www.arxiv.org/1804.11271>

From NNGP correspondence, empirically evaluates convergence rate and compare with Bayesian deep networks.

### Wide Neural Networks with Bottlenecks are Deep Gaussian Processes

<https://www.arxiv.org/abs/2001.00921>

In infinite network with bottleneck, which is some finite width hidden layers, the result is NNGP, which is composition of limiting GPs.

### Scale Mixtures of Neural Network Gaussian Processes

<https://arxiv.org/abs/2107.01408>

Show that simply introducing a scale prior on the last-layer parameters can turn infinitely wide neural networks of any architecture into a richer class of stochastic process, like heavy-tailed stochastic processes.

### alpha-Stable convergence of heavy-tailed infinitely-wide neural networks

<https://arxiv.org/abs/2106.11064>

Assuming that the weights of an MLP are initialized with i.i.d. samples from either a light-tailed or heavy-tailed distribution in the domain of attraction of a symmetric alpha-stable distribution for alpha in (0, 2]. Show that the vector of pre-activation values at all nodes of a given hidden layer converges in the limit, to a vecctor of i.i.d. random variables with symmetric alpha-stable distributions.

### On the expected behaviour of noise regularised deep neural networks as Gaussian processes

<https://www.arxiv.org/abs/1910.05563>

Consider impact of noise regularizations on NNGPs, and relate their behaviour to signal propagation theory.

### Infinitely Wide Graph Convolutional Networks: Semi-supervised Learning via Gaussian Processes

<https://www.arxiv.org/abs/2002.12168>

Inverstigate NNGP in GCNN, and propose a GP regression model with GCN, for graph-based semi-supervised learning.

### Deep Convolutional Networks as shallow Gaussian Processes

<https://www.arxiv.org/abs/1808.05587>

Show that the output of a CNN with an appropriate prior is a Gaussian Process in the infinite channel limit, and can be computed efficiently similar to a single forward pass through the original CNN with single filter per layer.

## The Limitations of Large Width in Neural Networks: A Deep Gaussian Process Perspective

<https://www.arxiv.org/abs/2106.06529>

Decouples capacity and width via the generalization of neural network to Deep Gaussian Process, aim to understand how width affects standard neural networks once they have sufficient capacity for a given modeling task. 

### Rate of Convergence of Polynomial Networks to Gaussian Processes

<https://arxiv.org/abs/2111.03175>

Demonstrate that the rate of convergence in 2-Wasserstein metric is O(sqrt(n)), where n is the number of hidden neurons in one hidden-layer neural network. Show the convergence rate for other atcivations, power-law for ReLU and inverse-sqrt for erf. 

## A self consistent theory of Gaussian Processes captures feature learning effects in finite CNNs

<https://www.arxiv.org/abs/2106.04110>

Consider the DNNs trained with noisy gradient descent on a large training set and derive a self consistent Gaussian Process theory accounting for strong finite-DNN and feature learning effects. 

### Deep Stable neural networks: large-width asymptotics and convergence rates

<https://arxiv.org/abs/2108.02316>

Show the deep stable neural network's weak convergence to stable stochastic process, with sup-norm convergence rate for joint growth and sequential growth. Show that joint growth leads to a slower rate than the sequential growth.

### On Connecting Deep Trigonometric Networks with Deep Gaussian Processes: Covariance, Expressivity, and Neural Tangent Kernel

<https://www.arxiv.org/abs/2203.07411>

Deep Gaussian Process with RBF kernel can be viewed as a deep trigonometric network with random feature layers and sine/cosine activation. In the wide limit with a bottleneck, show that the weight space view yield same effective covariance functions. Using this, DGPs ca be translated to deep trigonometric network which allows flexible and expressive prior distributions, and we can study DGP's neural tangent kernel.

### Normalization effects on shallow neural networks and related asymptotic expansions

<https://arxiv.org/abs/2011.10487>

Investigate the effect of normalization on shallow neural network in infinite width limit, show that to learning order in N, there is no bias-variance trade off and they both decreases.

### Infinite-channel deep stable convolution neural networks

<https://www.arxiv.org/abs/2102.03739>

Consider the problem of removing finite variance of infinite width network in convolutional NNs, show that the limit model is a stochastic process with multivariate stable finite dimensional distributions.

### Convergence of neural networks to Gaussian mixture distribution

<https://www.arxiv.org/abs/2204.12100>

Prove that fully connected deep neural network converge to a Gaussian mixture distribution with only last width goes to infinity.

### Avoiding Kernel Fixed Points: Computing with ELU and GELU Infinite Networks

<https://www.arxiv.org/abs/2002.08517>

Derive the infinite width kernel of MLPs with ELU and GELU and evaluate the performance of resulting GP. Analyze the fixed-point dynamics of iterated kernels for these activations, and show that unlike previous kernels, these kernels exhibit non-trivial fixed-point dynamics, which explains a mechanism for implicit regularisation in overparameterised deep models.

### Quantitative Gaussian Approximation of Randomly Initialized Deep Neural Networks

<https://www.arxiv.org/abs/2203.07379>

Derive the upper bound of the quadratic Wasserstein distance between randomly initialized neural network's output distribution and suitable Gaussian Process.

# NTK theory

### Neural tangent kernel: convergence and generalization in neural networks

<https://www.arxiv.org/abs/1806.07572>

Empirical kernel converges to deterministic kernel, and remains constant in 

### Wide neural networks of any depth evolve as linear models under gradient descent

<https://www.arxiv.org/abs/1902.06720>

Infinite neural network follows linear dynamics, and can be solved by solving linear ODE.

### Tensor Programs II: Neural Tangent Kernel for Any Architecture

<https://www.arxiv.org/abs/2006.14548>

Compute convergence to coordinate distribution for BP-like network, can use transpose. Can prove NTK convergence in initialization.

### Tensor Programs III: Neural Matrix Laws

<https://www.arxiv.org/abs/2009.10685>

Compute convergence to coordinate distribution, can use transpose. Can prove asymptotic freeness of matrices.

### Tensor Programs IIb: Architectural Universality of Neural Tangent Kernel Training Dynamics

<https://www.arxiv.org/abs/2105.03703>

Prove NTK convergence in training. Derive back propagation as a vector in program.

### Feature Learning in Infinite-Width Neural Networks

<https://www.arxiv.org/abs/2011.14522>

Define feature learning in terms of asymptotic size of output, design new parametrization.

### Mean-field Behaviour of NTK

<https://www.arxiv.org/abs/1905.13654>

Bridge the gap between NTK theory and EOC initialization.

### On Random Kernels of Residual Architectures

<https://www.arxiv.org/abs/2001.10460>

Study finite width and depth corrections for the NTK of ResNets and DenseNets. 
Finite size residual architecture are initialized much closer to the kernel regime than vanilla.

### The recurrent neural tangent kernel

<https://www.arxiv.org/abs/2006.10246>

Study NTK for recurrent neural networks. 

### Infinite Attention: NNGP and NTK for Deep Attention Networks

<https://www.arxiv.org/abs/2006.10540>

Study NTK for attention layers, and propose modifiationc of the attention mechanism.

### Scaling limits of wide neural networks with weight sharing: Gaussian process behavior, gradient independence, and neural tangent kernel derivation

<https://www.arxiv.org/abs/1902.04760>

The very beginning of tensor program. 

### Finite depth and width corrections to the neural tangent kernel

<https://www.arxiv.org/abs/1909.05989>

Prove the precise scaling for the mean and variance of the NTK. When both depth and width tends to infinity, NTK is no more deterministic with non-trivial evolution.

### Dynamics of deep neural networks and neural tangent hierarchy

<https://www.arxiv.org/abs/1909.08156>

Study NTK dynamics for finite width deep FCNNs. Derive an infinite hierarchy of ODEs, the neural tangent hierarchy which captures the gradient descent dynamic.
The truncated hierarchy can approximate the dynamic of the NTK up to arbitrary precision.

### On the neural tangent kernel of deep networks with orthogonal initialization

<https://www.arxiv.org/abs/2004.05867>

Study dynamics of ultra-wide networks including FCN, CNN with orthogonal initialization via NTK. 
Prove that Gaussian weight and orthogonal weight's NTK are equal in infinite width, and both stays constant. 
It suggests that orthogonal initialization does not speed up training. 

### On Exact Computation with an Infinitely Wide Neural Net

<https://www.arxiv.org/abs/1904.11955>

First efficient exact algorithm for computing the extension of NTK to CNN, as well as an efficient GPU implementation. 

### Beyond Linearization: On Quadratic and Higher-Order Approximation of Wide Neural Networks

<https://www.arxiv.org/abs/1910.01619>

Investigate the training of over-parametrized NNs that are beyoung the NTK regime, yet still governed by the Taylor expansion.

### Towards Understanding Hierarchical Learning: Benefits of Neural Representations

<https://www.arxiv.org/abs/2006.13436>

Using random wide two-layer untrainable networks as a representation function, if the trainable network is the quadratic Taylor model of a wide two-layer network,
neural representation can achieve improved sample complexities. But this does not increase in NTK.

### Neural Tangents: Fast and Easy Infinite Neural Networks in Python

<https://www.arxiv.org/abs/1912.02803>

High level API for specifying complex and hierarchical neural network architectures. 

### On the infinite width limit of neural networks with a standard parameterization

<https://www.arxiv.org/abs/2001.07301>

Propose an imporved extrapolation of the standard parameterization that yields a well-defined NTK. 

### Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Models

<https://www.arxiv.org/abs/1905.13192>

Presents a new class of graph kernel, Graph Neural Tangent Kernels which corerespond to infinitely wide multi-layer GNNs trained by gradient descent.
GNTKs provably learn a class of smooth functions on graphs.

### On the Inductive Bias of Neural Tangent Kernels

<https://www.arxiv.org/abs/1905.12173>

Study inductive bias of learning in NTK regime by analyzing the kernel and RKHS. Specifically, study smoothness, approximation, stability properties with finite norm.

### A Fine-Grained Spectral Perspective on Neural Networks

<https://www.arxiv.org/abs/1907.10599>

Study the spectra of NNGP and NTK kernel, show insights on some questions on neural networks.

### Scaling description of generalization with number of parameters in deep learning

<https://www.arxiv.org/abs/1901.01608>

Show that conflict of result about generalization error in over-parameterization using NTK, in specific that the initialization causes finite-size random fluctuations
around its expectation.

### Disentangling Trainability and Generalization in Deep Neural Networks

<https://www.arxiv.org/abs/1912.13053>

By analyzing the spectrum of the NTK, formulate necessary conditions for trainability and generalization across a range of architectures. 

### Enhanced Convolutional Neural Tangent Kernels

<https://www.arxiv.org/abs/1911.00809>

Modify the kernel using a new operation called Local Average Pooling, and show that Global Average Pooling is equivalent to full translation data augmentation.

### Learning and Generalization in Overparameterized Neural Networks, Going Beyond Two Layers

<https://www.arxiv.org/abs/1811.04918>

Prove taht overparameterized neural networks can learn some notable concept classes, and can be done by SGD in polynomial time. Uses a new notion of quadratic approximation of NTK, and connect it to the SGD theory of escaping saddle points.

### On the Random Conjugate Kernel and Neural Tangent Kernel

<http://proceedings.mlr.press/v139/hu21b.html>

Derive the distribution and moment of diagonal elements of kernel, for feedforward network with random initialization and residual network with infinite branches.

### Spectra of the Conjugate Kernel and Neural Tangent Kernel for linear-width neural networks

<https://www.arxiv.org/abs/2005.11879>

Study eigenvalue distributions for NN kernels, show that they converges to deterministic limits, each explained by Marcenko-Pastur map and linear combination of those.

### Deep Networks and the Multiple Manifold Problem

<https://www.arxiv.org/abs/2008.11245>

Given two low-dimensional submanifold of the unit sphere, show that given polynomially many samples w.r.t. depth, the feedforward network can perfectly classify manifolds,
using NTK in the nonaymptotic analysis of training.

### Weighted Neural Tangent Kernel: A Generalized and Improved Network-Induced Kernel

<https://www.arxiv.org/abs/2103.11558>

Introduce weighted NTK, that can capture the training dynamics for different optimizers, and prove the stability of WNTK and equivalence betwen WNTK regression estimator and the corresponding NN estimator.

### Neural tangent kernels, transportation mappings, and universal approximation

<https://arxiv.org/abs/1910.06956>

Provides a generic scheme to aproximate functions with the NTK by sampling from transport mappings, and the construction of transport mappings via Fourier transforms

### On the Similarity between the Laplace and Neural Tangent Kernels

<https://arxiv.org/abs/2007.01580>

Theoretically show that for normalized data on the hypersphere, both NTK and Laplace kernel have the same eigenfunctions and their eigenvalues decay polynomially at the same rate, implying that their RKHS are equal, and they share same smoothness properties.

### Deep Neural Tangent Kernel and Laplace Kernel Have the Same RKHS

<https://arxiv.org/abs/2009.10683>

Prove that RKHS of NTK and the Laplace kernel include the same set of functions, and show that the exponential power kernel with a smaller power leads to a larger RKHS.

### Convergence of Adversarial Training in Overparametrized Neural Networks

<https://arxiv.org/abs/1906.07916>

Show that the adversarial training converges to a network where the surrogate loss w.r.t. attack algorithm has small optimal robust loss, and show that the optimal robust loss is also close to zero, giving robust classifier.

### A Deep Conditioning Treatment of Neural Networks

<https://arxiv.org/abs/2002.01523>

Show that depth improves trainability of NNs by improving the conditioning of certain kernel matrices of the input data. 

### Towards an Understanding of Residual Networks Using Neural Tangent Hierarchy (NTH)

<https://arxiv.org/abs/2007.03714>

Study dynamics of the NTK for finite width ResNet using the NTH, reducing the requirement on the layer width w.r.t. number of training samples from quartic to cubic.

### Scalable Neural Tangent Kernel of Recurrent Architectures

<https://arxiv.org/abs/2012.04859>

Extend the family of kernels associated with RNNs, to more complex architectures like bidirectional RNNs and average pooling. Also develop a fast GPU implementation for these.

### The Surprising Simplicity of the Early-Time Learning Dynamics of Neural Networks

<https://arxiv.org/abs/2006.14599>

Prove that for a class of well-behaved input distributions, the early-time learning dynamics of a two-layer fully-connected neural network can be mimicked by training a simple linear model on the inputs, by bounding the spectral norm of the difference between the NTK at init and an affine transform of the data kernel, while allowing the network to escape the kernel regime later.

### Benefits of Jointly Training Autoencoders: An Improved Neural Tangent Kernel Analysis

<https://arxiv.org/abs/1911.11983>

Prove the linear convergence of gradient descent in two learning regimes, only encoder is trained or jointly trained, in two-layer over-parameterized autoencoders, giving the considerable benefits of joint training over weak training.

### Neural Tangent Kernel Maximum Mean Discrepancy

<https://arxiv.org/abs/2106.03227>

Present NN MMD statistic by identifying connection between NTK and MMD statics, allowing us to understand the properties of the new test statistic like Type-1 error and testing power.

### A Neural Tangent Kernel Perspective of GANs

<https://arxiv.org/abs/2106.05566>

Use NTK on GAN, show that GAN trainability primarily depends on the discriminator's architecture.

### The Neural Tangent Kernel in High Dimensions: Triple Descent and a Multi-Scale Theory of Generalization

<https://arxiv.org/abs/2008.06786>

Provide a precise high-dimensional asymptotics analysis of generalization under kernel regression with NTK, and show that the test error has non-monotonic behavior in the overparameterized regime.

### Optimal Rates for Averaged Stochastic Gradient Descent under Neural Tangent Kernel Regime

<https://arxiv.org/abs/2006.12297>

Show that the averaged stochastic gradient descent can achieve the minimax optimal convergence rate, with the global convergence guarantee, by exploiting the complexities of the target function and the RKHS of NTK. 

### Tight Bounds on the Smallest Eigenvalue of the Neural Tangent Kernel for Deep ReLU Networks

<https://arxiv.org/abs/2012.11654>

Smallest eigenvalue of the NTK has been related to the memorization capacity, the global convergence, and the generalization. Provide tight bounds on the smallest eigenvalue of NTK matrices for deep ReLU nets, both infinite and finite width.

### A Generalized Neural Tangent Kernel Analysis for Two-layer Neural Networks

<https://arxiv.org/abs/2002.04026>

Provide a generalized NTK analysis and show that noisy gradient descent with weight decay can still exhibit a kernel-like behavior.

### On the Benefit of Width for Neural Networks: Disappearance of Bad Basins

<https://www.arxiv.org/abs/1812.11039>

There is a phase transition from having sub-optimal basins to no sub-optimal basins in wide network. On positive side, for any continuous activation functions, the loss surface has no sub-optimal basins, on the negative side, for a large class of networks with width below a threshold, there is a strict local minima that is not global.

### Quantum-enhanced neural networks in the neural tangent kernel framework

<https://www.arxiv.org/abs/2109.03786>

Study a class of QCNN where NTK theory can be directly applied. 

### A Neural Tangent kernel Perspective of Infinite Tree Ensembles

<https://www.arxiv.org/abs/2109.04983>

Considering an ensemble of infinite soft trees, introduce Tree Neural Tangent Kernel, find some nontrivial properties like effect of oblivious tree structure and the degeneracy of the TNTK induced by the deepening of the trees.

### The Interpolation Phase Transition in Neural Networks: Memorization and Generalization under Lazy Training

<https://www.arxiv.org/abs/2007.12826>

Characterize the eigenstructure of the empirical NTK in the overparameterized regime, which implies that minimum eigenvalue of the empirical NTK is bounded away from zero. 
And also prove that the test error is well approximated by the one of kernel ridge regression w.r.t. the infinite-width kernel.

### Uniform Generalization Bounds for Overparameterized Neural Networks

<https://www.arxiv.org/abs/2109.06099>

Using NTK theory, prove uniform generalization bounds in kernel regime, when the true data generating model belongs to the RKHS corresponding to the NTK. 

### Understanding neural networks with reproducing kernel Banach spaces

<https://www.arxiv.org/abs/2109.009710>

Prove a representer theorem for a wide class of reproducing kernel Banach spaces that admit a suitable integral representation and include one hidden layer neural networks of possibly infinite width. And show that the norm in the RKBS can be characterized in terms of the inverse Radon transform of a bounded real measures.

### Defomed semicircle law and concentration of nonlinear random matrices for ultra-wide neural networks

<https://www.arxiv.org/abs/2109.09304>

Obtain the limiting spectral distributions of CK and NTK, in the ultra width regime, a defomed semicircle law appears.

### Predicting the outputs of finite deep neural networks trained with noisy gradients

<https://arxiv.org/abs/2004.01190>

In the infinite-width limit, establish a correspondence between DNNs with noisy gradients and the NNGP, provide a general analytical form for the finite width corrections with predicion of outputs, finally flesh out algebraically how these FWCs can improve the performance of finite convolutional neural networks relative to their GP counterparts.

### Regularization Matters: A Nonparametric Perspective on Overparametrized Neural Network

<https://arxiv.org/abs/2007.02486>

Prove that for overparametrized one-hidden-layer ReLU neural networks with l2 regularization, the output is close to that of the kernel ridge regression with the corresponding neural tnagent kernel, and minimax optimal rate of L2 estimation error can be achieved.

### On the Provable Generalization of Recurrent Neural Networks

<https://arxiv.org/abs/2109.14142>

Using detailed analysis about the NTK matrix, prove a generalization error bound to learn such functions without normalized conditions and show that some notable concept classes are learnable with the numbers of iterations and samples scaling almost-poynomially in the input length L. 

### Order and Chaos: NTK views on DNN Normalization, Checkerboard and Boundary Artifacts

<https://arxiv.org/abs/1907.05715>

Show the order and chaos regime, and show that scaled ReLU gives ordered regime, layer normalization and batch normalization leads chaotic regime, which also appears in CNN. Analysis explains so called checkerboard patterns and border artifacts, with proposal of methods removing these effects.

### Label Propagation across Graphs: Node Classification using Graph Neural Tangent Kernels

<https://arxiv.org/abs/2110.03763>

In the inductive setting of node classification, where the unlabeled target graph is completely separate, that there are no connections between labeled and unlabeled nodes, using GNTK to find correpondences between nodes in different graphs.

### New Insights into Graph Convolutional Networks using Neural Tangent Kernels

<https://arxiv.org/abs/2110.04060>

Using the NTK, try to explain the behavior that performance of GCNs degrades with increasing network depth, and it improves marginally with depth using skip connections. Identify that with suitable normalization, network depth does not always drastically reduce the performance of GCNs.

### On the Convergence and Calibration of Deep Learning with Differential Privacy

<https://arxiv.org/abs/2106.07830>

Precisely characterize the effect of per-sample clipping on the NTK matrix and show that the noise level of DP optimizers does not affect the convergence in the gradient flow regime. In particular, the local clipping breaks the positive semi-definiteness of NTK, which can be preserved by the global clipping. 

### DNN-Based Topology Optimisation: Spatial Invariance and Neural Tangent Kernel

<https://arxiv.org/abs/2106.05710>

Study the Solid Isotropic Material Penalisation method with a desntiy field generated by a FCNN. Show that the use of DNNs leads to a filtering effect similar to traditional filtering techniques for SIMP in th large width limit, with a filter described by the NTK.

### Neural Tangent Kernel Eigenvalues Accurately Predict Generalization

<https://arxiv.org/abs/2110.03922>

We extend recent results to demonstrate that, by examining the eigensystem of a neural network's "neural tangent kernel", one can predict its generalization performance when learning arbitrary functions, not only mean-squared-error but all first and second-order statistics of learned function. Also prove a new NFL theorem characterizing a fundamental tradeoff in the inductive bias of wide neural networks: improving a network's generalization for a given target function must worsen its generalization for orthogonal functions.

### The Local Elasticity of Neural Networks

<https://arxiv.org/abs/1910.06943>

A classifier is said to be locally elastic if its prediction at a feature vector is not significantly perturbed after the classifier is updated via SGD. Offer a geometric interpretation of local elasticity using NTK, and obtain pairwise similarity measures between feature vectors.

### Deformed semicircle law and concentration of nonlinear random matrices for ultra-wide neural networks

<https://arxiv.org/abs/2109.09304>

Obtain the limiting spectra distributions of conjugate kernel and NTK in two-layer fully connected networks. Under the ultra-width regime, a deformed semicircle law appears. Also prove non-asymptotic concentrations of of empirical CK and NTK around their limiting kernel in the spectra norms.

### Enhanced Recurrent Neural Tangent Kernels for Non-Time-Series Data

<https://arxiv.org/abs/2012.04859>

Extend the family of kernels associated with RNNs, to more complex architectures including bidirectional RNNs and RNNs with average pooling.

### How Neural Networks Extraploate: From Feedforward to Graph Neural Networks

<https://arxiv.org/abs/2009.11848>

Quantify the observation that ReLU MLPs quickly converge to linear functions along any direction from the origin, which implies that ReLU MLPs do not extrapolate most nonlinear functions. And show that the success of GNNs in extrapolating algorithmic tasks to new data relies on encoding task-specific non-linearities in the architecture or features. Theoretical analysis builds on a connection of over-parameterized networks to the NTK.

### SGD Learns the Conjugate Kernel Class of the Network

<https://www.arxiv.org/abs/1702.08503>

Show that SGD is guaranteed to learn a function that is competitive with the best function in the conjugate kernel space, in polynomial time.

### Implicit Acceleration and Feature Learning in Infinitely Wide Neural Networks with Bottlenecks

<https://www.arxiv.org/abs/2107.00364>

Analyze the learning dynamics of infinitely wide neural networks with a finite sized bottlenecks. This allows data dependent feature learning in its bottleneck representation, unlike NTK limit.

### Tighter Sparse Approximation Bounds for ReLU Neural Networks

<https://arxiv.org/abs/2110.03673>

Derive sparse neural network approximation bounds that refine previous works, and show that infinite-width neural network representations on bounded open sets are not unique.

### Deep Networks Provably Classify Data on Curves

<https://arxiv.org/abs/2107.14324>

Prove that when the network depth is large relative to geometric properties and the network width and number of samples is polynomial in depth, randomly initialized gradient descent quickly learns to correctly classify all points on the two curves with high probability. Analyze by a reduction to dynamics in the NTK regime, using the fine-grained control of the decay properties, showing that NTK can be locally approximated by a translationally invariance operator on the manifolds and stable inverted over smooth functions, which guarantees convergence and generalization.

### Training Integrable Parameterizations of Deep Neural Networks in the Infinite-Width Limit

<https://arxiv.org/abs/2110.15596>

Study a specific choice of small initialization corresponding to mean-field limit, calling integrable parameterization. Show that under standard i.i.d. zero-mean init, integrable parameterization with more than four layers start at a stationary point in the infinite-width limit with no learning. Then propose various methods to avoid this triviality, for example using large initial learning rates which is equivalent to maximal update parameterization.

### Neural Networks as Kernel Learners: The Silent Alignment Effect

<https://arxiv.org/abs/2111.00034>

Demonstrate that neural networks in the rich feature learning regime can learn a kernel machine with a data-dependent kernel, due to a phenomenon termed silent alignment, requiring that the NTK evolves in eigenstructure while small and before the loss decreases, and grows noly in overall scale afterwards. Show that such an effect takes place in homogeneour NNs with small initialization and whitened data. 

### Quantifying the generalization error in deep learning in terms of data distribution and neural network smoothness

<https://arxiv.org/abs/1905.11427>

Introduce the cover complexity to measure the difficulty of a data set and the inverse of the modulus of continuity to quantify neural network smoothness, deriving quantitative bound for expected error.

### Understanding Layer-wise Contributions in Deep Neural Networks through Spectral Analysis

<https://arxiv.org/abs/2111.03972>

Analyze the layer-wise spectral bias of DNNs and relate it to the contributions of different layers in the reduction of generalization error for a given target function. Using Hermite polynomials and spherical harmonics, prove that initial layers exhibit a larger bias towards high-frequency functions defined on the unit sphere.

### When Do Neural Networks Outperform Kernel Methods?

<https://arxiv.org/abs/2006.13409>

Show that the curse of dimensioanlity of RKHS methods becomes milder if the covaraiates display the same low-dimensional structure as the target function, and we precisely characterize this tradeoff. 

## On the Equivalence between Neural Network and Support Vector Machine

<https://arxiv.org/abs/2111.06063>

Propose the equivalence between NN and SVM, especially infinite NNs trained by soft margin loss and soft margin SVM with NTK. Show that every finite-width NN with regularized loss functions is approximately a kernel machine, with generalization bound for NN using the kernel machine, robustness certificate for infinite-width NNs, instrinsically more robust infinite-width NNs.

### Towards Understanding the Condensation of Neural Networks at Initial Training

<https://arxiv.org/abs/2105.11686>

Empirical works show that input weights of hidden neurons condense on isolated orientation with a small initialization. 

### Critical initialization of wide and deep neural networks through partial Jacobians: general theory and applications to LayerNorm

<https://arxiv.org/abs/2111.12143>

Describe a new way to diagnose criticality of NN, by partial Jacobians, which is a derivative of preacitvations in layer l for earlier layers, and discuss various properties of the partial Jacobians such as scaling and relation to NTK. Using the recurrence relation for partial Jacobian, analyze the criticality of deep MLP, with/without LayerNorm.

### Neural Optimization Kernel: Towards Robust Deep Learning

<https://arxiv.org/abs/2106.06097>

Establish the connection between DNN and kernel family Neural Optimization Kernel. NOK performs monotonic descent updates of implicit regularization problems, and can implicitly choose by different activation functions, establishing a new generalization bound.

### Neural Tangent Kernel of Matrix Product States: Convergence and Applications

<https://arxiv.org/abs/2111.14046>

Study the NTK of matrix product states and the convergence, show that NTK of MPS asymptotically converges to a constant matrix as the bond dimension of MPS go to infinity. 

### Understanding Square Loss in Training Overparametrized Neural Network Classifiers

<https://arxiv.org/abs/2112.03567>

Contribute to the theoretical understanding of square loss in classification by how it performs for overparameterized neural networks in the NTK regime. When non-separational case, fast convergence rate is established for both misclassification rate and calibration error, and separation case exponentially fast rate. Also prove lower bounded margin.

## On Lazy Training in Differentiable Programming

<https://www.arxiv.org/abs/1812.07956>

Show that lazy training phenomenon is due to a choice of scaling, that makes the model behave as its linearization around the initialization.

### Taylorized Training: Towards Better Approximation of Neural Network Training at Finite Width

<https://www.arxiv.org/abs/2002.04010>

Use k-th Taylor expansion of the neurl network at initialization, and show that the approximation error decay exponentially over k in wide neural networks.

### On the Provable Generalization of Recurrent Neural Networks

<https://arxiv.org/abs/2109.14142>

Prove a generalization error bound using NTK analysis, for the RNN withtout normalization, for additive concept class and N-variables functions.

### Eigenspace Restructuring: a Principle of Space and Frequency in Neural Networks

<https://arxiv.org/abs/2112.05611>

Show that the topologies from deep CNNs restructure the associated eigenspaces into finer subspaes, then MLPs. This new structure also depends on the concept class, measuring the spatial distance among nonlinear interaction terms, and this analysis improves the network's learnability. Finally prove a sharp characterization of generalization error for infinite width CNNs.

### Rethinking Influence Functions of Neural Networks in the Over-parameterized Regime

<https://arxiv.org/abs/2112.08297>

Utilize NTK theory to calculate influence function for network trained with regularized mean-squared loss, proving that approximate error can be arbitrarily small, analyze the error bound for classic IHVP method, which depends on regularization term and probability density of corresponding training points.

### Training Integrable Parameterization of Deep Neural Networks in the Infinite-Width Limit

<https://arxiv.org/abs/2110.15596>

Study integrable parameterization, which corresponds to mean-field limit, show that with more than four layers the initialization is already stationary and no learning occurs. Then propose methods to escape this behavior, like large learning rate, which is equivalent to maximal update parameterization.

### Infinite width (finite depth) neural networks benefit from multi-task learning unlike shallow Gaussian Processes -- an exact quantitative macroscopic characterization

<https://www.arxiv.org/abs/2112.15577>

Provide that optimizing large ReLU NNs with at least one hidden layer with L2 regularization enforces representation learning even in infinite width limit.

### Deep Neural Networks as Point Estimates for Deep Gaussian Processes

<https://www.arxiv.org/abs/2105.04504>

Establish an equivalence between forward pass of neural network and deep/sparse Gaussian process, which interprets activationn as interdomain inducing featrues through analysis of the interplay between activation and kernels. 

### Generalization Bounds of Stochastic Gradient Descent for Wide and Deep Neural Networks

<https://www.arxiv.org/abs/1905.13210>

Show that expected 0-1 loss of wide enough ReLU networks with SGD training and random initialization can be bounded by the training loss of random feature model induced by gradient at initilization, named neural tangent random feature model. Establish strong connection between generalization error bound and NTK.

### Why bigger is not always better: on finite and infinite neural networks

<https://www.arxiv.org/abs/1910.08013>

Give analytic results characterising the prior over representations and representation learning in finite deep linear newtorks, unlike infinite networks which fails to learn representation.

### Largest Eigenvalues of the Conjugate Kernel of Single-Layered Neural networks

<https://www.arxiv.org/abs/2201.04753>

Show that the largest eigenvalue of f(WX) where W=YY^T, has the same limit as that of some well-known linear random matrix ensembles. Relate this limit to that of information+noise random matrix, 
showing phase transition depending on activation, W, X.

### Implicit Bias of MSE Gradient Optimization in Unparameterized Neural Networks

<https://www.arxiv.org/abs/2201.04738>

Study the dynamics of a NN in function space when optimizing the mean squared error via gradient flow, show that in the underparameterized regime, the network learns eigenfunctions of an integral operator determined by the NTK. For S^d-1 data distribution and rotation invariant weight distribution, the eigenfunction is spherical harmonics, showing spectral bias in the underparameterized regimes.

### Kernel Methods and Multi-layer Perceptrons Learn Linear Models in High Dimensions

<https://www.arxiv.org/abs/2201.08082>

Show that for a large class of kernels including NTK, kernel methods can only perform as well as linear models in high dimensional regime where covariates are independent. Also show that when the data is generated by a kernel method where the relationship is very nonlinear, the linear model are in fact optimal, among all models including nonlinear models. 

### An Infinite-Feature Extension for Bayesian ReLU Nets That Fixes Their Asymptotic Overconfidence

<https://www.arxiv.org/abs/2010.02709>

Show that the infinite ReLU features obtained by infinite width limit is asymptotically maximally uncertain far away from the data while predictive power is unaffected near the data.

### Stochastic Neural Networks with Infinite Width are Deterministic

<https://www.arxiv.org/abs/2201.12724>

Prove that as the width of an optimized stochastic neural networks tends to infinity, predictive variance decreases to zero, with application on dropout and variational autoencoder.

### Reverse Engineering the Neural Tangent Kernel

<https://www.arxiv.org/abs/2106.03186>

Prove that any positive-semidefinite dot-product kernel can be realized as either conjugate or NTK of a shallow neural network, with an approapriate choice of activation function.

### Spectral Bias and Task-Model Alignment Explain Generalization in Kernel Regression and Infinitely Wide Neural Networks

<https://www.arxiv.org/abs/2006.13198>

Derive analytical expression for generalization error applicable to any kernel and data distribution, including those arising from infinite neural networks.

### Double-descent curves in neural networks: a new perspective using Gaussian Processes

<https://www.arxiv.org/abs/2102.07238>

Using NNGP and random matrix theory, argue that this effict is explained by convergence to limiting Gaussian processes.

### Embedded Ensembles: Infinite Width Limit and Operating Regimes

<https://www.arxiv.org/abs/2202.12297>

Analyze Embedded ensembles, like Batch Ensemble or Monte-Carlo dropout ensembles, which is ensemble method that most of weights are shared by means of a single reference network. Using NTK approach, show that there is two ensemble regime, independent and collective, and show that independent regime behaves as an ensemble of independent models.

### The Spectral Bias of Polynomial Neural Networks

<https://www.arxiv.org/abs/2202.13473>

Conduct a spectral analysis of the NTK of polynomial neural networks, and find that 2-Net family sppeds up the learning of the higher frequencies.

### On the linearity of large non-linear models: when and why the tangent kernel is constant

<https://www.arxiv.org/abs/2010.01092>

Show that constancy of tangent kernel depends on infinity norm and l2 norm, which controls the gradient and hessian of network. Show that even with lazy training effect, the network with nonlinear output and network with bottlneck has non-constant tangent kernel.

### Transition to Linearity of Wide Neural Networks is an Emerging Property of Assembling Weak Models

<https://www.arxiv.org/abs/2203.05104>

Provide a new perspective that the transition to lienarity is property of assembly model, where neural network can viewed as assembly model recursively build from sub-models, which is neurons of former layers.

### On the Spectral Bias of Convolutional Neural Tangent and Gaussian Process Kernels

<https://www.arxiv.org/abs/2203.09255>

Prove that eigenfunctions of CNN's NTK and NNGP kernel with the uniform measure are formed by products of spherical harmonics, defined over the channels of the different pixels. Then use hierarchical factorizable kernels to bound the eigenvalues, and show that eigenvalues decay polynomially and derive measures that reflect the composition of hierarchical features in these networks.

### On the Neural Tangent Kernel Analysis of Randomly Pruned Wide Neural Networks

<https://www.arxiv.org/abs/2203.14328>

Show that randomly pruned FCNN's empirical NTK converges to original NTK with some scaling factor, which can be removed by adding scaling after pruning. Using this, give a non-asymptotic bound on the approximation error in terms of pruning probability.

### Neural Q-Learning for solving elliptic PDEs

<https://www.arxiv.org/abs/2203.17128>

Develop a new numerical method for solving elliptic type PDEs by adapting the Q-learning, and using NTK approach, prove that the neural network converges, and for monotone PDE, despite the lack of a spectral gap in the NTK, prove that limit NN converges in L2 to the PDE solution.

### Memorization and Optimization in Deep Neural Networks with Minimum Over-parameterization

<https://www.arxiv.org/abs/2205.10217>

Show a lower bound on the smallest NTK eigenvalue with the minimum possible over-parameterization with sqrt(N) width, and provide the memorization capacity and the optimization guarantee using this bound.

### A Neural Tangent Kernel Formula for Ensembles of Soft Trees with Arbitrary Architectures

<https://www.arxiv.org/abs/2205.12904>

Formulate and analyze the NTK induced by soft tree ensembles for arbitrary tree architectures, and show that only the number of leaves at each depth is relevant with infinitely many trees.
Also show that the NTK of asymmetric trees does not degenerate with infinite depth, which is contrast to binary tree.

### Spectral Bias Outside the Training Set for Deep Networks in the Kernel Regime

<https://www.arxiv.org/abs/2206.02927>

Provide quantitative bounds measuring the L2 distance in function space between the trajectory of a finite-width network from the idealized kernel dynamics of infinite width and data. The bound imply that the network is biased to learn the top eigenfunctions of the NTK no just on the training set but over the entire input space.

### The Interpolation Phase Transition in Neural Networks: Memorization and Generalization under Lazy Training

<https://www.arxiv.org/abs/2007.12826>

Characterize the eigenstructure of empirical NTK in the overparameterized regime, which implies that the minimum eigenvalue is bounded away from zero, and therefore the network can exactly interpolate arbitrary labels in the same regime.
Then characterize the generalization error of ridge regression, prove that the test error is well apprximated by the one of kernel ridge regression with respect to the infinite-width kernel.

### How infinitely Wide Neural Networks Benefit from Multi-task Learning -- an Exact Macroscopic Characterization

<https://www.arxiv.org/abs/2112.15577>

Prove that optimizing wide ReLU neural networks with at least one hidden layer using L2-regularization on the paramters enforces multi-task learning due to representaion-learning, even in the limiting regime.

# Infinite Depth

### The Neural Covariance SDE: Shaped Infinite Depth-and-Width Networks at Initialization

<https://www.arxiv.org/abs/2206.02766>

Study the random covariance matrix of logits which is defined by the penultimate layer's activation, in infinite depth-and-width limit to consider the fluctuation accumulated over the layers. Identify the scaling of activation that this do not arrive at trivial limit, and show that this matrix is governed by a SDE named Neural Covariance SDE. 

## The Future is Log-Gaussian: ResNets and Their Infinite-Depth-and-Width Limit at Initialization

<https://www.arxiv.org/abs/2106.04013>

Show that the ReLU ResNets exhibits log-Gaussian behaviour at initialization in the infinite-depth-and-width limit, with parameters depending on the ratio d/n. Show that ReLU ResNet is hypoactivated, that fewer than half of the ReLUs are activated.

### Why Do Deep Residual Networks Generalize Better than Deep Feedforward Networks? -- A Neural Tangent Kernel Perspective

<https://www.arxiv.org/abs/2002.06262>

Show that training ResNets can be viewed as learning reproducing kernel functions with some kernel function. THen compare the kernel of two networks, and show that the class of functions induced by FFNets is asymptotically not learnable, which does not happens in ResNets.

### Neural Tangent Kenrel Beyond the Infinite-Width Limit: Effects of Depth and Initialization

<https://www.arxiv.org/abs/2202.00553>

Study NTK of ReLU NNs with depth comparable to width, proving that the NTK properties depend significantly on the depth-to-width ratio and initialization. This indicate the importance of ordered/chaotic regime and edge of chaos. Show that NTK variability grows exponentially at EOC and chaotic phase, but not in ordered phase. Also show that NTK of deep networks in ordered phase may stay constant during training.

### Neural Tangent Kernel Analysis of Deep Narrow Neural Networks

<https://www.arxiv.org/abs/2202.02981>

Study infinite depth limit of MLP with specific initialization, and establish a trainability guarantee with NTK theory.

### Wide and Deep Neural Networks Achieve Optimality for Classification

<https://www.arxiv.org/abs/2204.14126>

Analyze infinitely wide and infinitely deep neural networks, and using the connection to NTK, provide explicit activation functions that are optimal in the sense of misclassification. Create a taxonomy of infinite networks and show that these models implement on of three classifiers depending the activation function, 1) 1-nearest neighbor, 2) majority vote, 3) singular kernel classfiers.

### Exact Convergence Rates of the Neural Tangent Kernel in the Large Depth Limit

<https://www.arxiv.org/abs/1905.13654>

Provide a comprehensive analysis of the convergence rates of the NTK regime to the infinite depth regimes.

### Neural tangent kenrel analysis of shallow alpha-Stable ReLU neural networks

<https://arxiv.org/abs/2206.08065>

Consider the NTK of alpha-stable NNs, showing that their training is equivalent to a kernel regression with an alpha/2-stable random kernel.

# Application

### Finding sparse trainable neural networks through Neural Tangent Transfer

<https://www.arxiv.org/abs/2006.08228>

Introduce Neural Tangent Transfer, a method that finds trainable sparse networks in a label-free manner, that whose training dynamics computed by NTK is similar to dense ones.

### Rapid training of deep neural networks without skip connections or normalization layers using Deep Kernel Shaping

<https://arxiv.org/abs/2110.01765>

Using NTK theory and Q/C map analysis, identify the main pathologies in deep networks that prevent them from training fast and generalizing to unseen data, and show how these can be avoided by carefully controlling the shape of the network's initialization-time kernel function. Develop a method called DKS, which accomplishes this using a combination of precise parameter initilization, activation function transformation, and small architectural tweaks. 

### Meta-Learning with Neural Tangent Kernels

<https://www.arxiv.org/abs/2102.03909>

Generalize MAML to function space, eliminating need of sub-optimal iterative inner-loop adaption by replacing the adaption with a fast-adaptive regularizer in the RKHS and solving the adaption analytically based on the NTK theory.

### FL-NTK: A Neural Tangent Kernel-based Framework for Federated Learning Convergence Analysis

<https://arxiv.org/abs/2105.05001>

Presents a new class of convergence analysis for federated learning, which corresponds to overparameterized ReLU NNs trained by gradient descent in FL. Theoretically FL-NTK converges to a global optimal solution at a linear rate, and also achieve good generalizations.

### Learning with Neural Tangent Kernels in Near Input Sparsity Time

<https://www.arxiv.org/abs/2104.00415>

Accelerate kenrel machines with NTK, by mapping the input data to a randomized low-dimensional feature space so that the inner product of the transformed data approximates the NTK evaluation, based on polynomial expansion of arc-cosine kernels.

### Bayesian deep ensembles via the Neural tangent kernel

<https://www.arxiv.org/abs/2007.05864>

Using the NTK, add bias to initialization derive bayesian interpretation for deep ensembles.

### Exact marginal prior distributions of finite Bayesian neural networks

<https://arxiv.org/abs/2104.11734>

Derive exact solutions for the function space priors for individual input examples of a class of finite fully-connected feedforward Bayesian neural networks. Deep linear networks have prior as a simple expression in terms of the Meijer G-function. The prior of a finite-ReLU network is a mixture of the priors of linear networks of smaller widths.

### Neural Architecture Search on ImageNet in Four GPU Hours: A Theoretically Inspired Perspective

<https://arxiv.org/abs/2102.11535>

By analyzing the spectrum of the NTK and the number of linear regions in the input spaces, show that these two measurements imply the trainability and expressivity of a neural network and they strongly correlate with the network's test accuracy. 

### Neural Tangent Kernel Empowered Federated Learning

<https://arxiv.org/abs/2110.03681>

Propose a novel FL paradigm empowered by the NTK framework, which addresses the challenge of statistical heterogenity by transmitting update data that are more expressive than those of the traditional FL paradigms.

### An Infinite-Feature Extension for Bayesian ReLU Nets That Fixes Their Asymptotic Overconfidence

<https://arxiv.org/abs/2010.02709>

Extend finite BNNs with infinite ReLU features via the GP, showing that the resulting model is asymptotically maximally uncertain far away from the data, while the BNN's predictive power is unaffected near the data.

### DNN-Based Topology Optimisation: Spatial Invariance and Neural Tangent Kernel

<https://arxiv.org/abs/2106.05710>

Show that the use of DNN in Solid Isotropic Material Penalisation leads to a filtering effect with filter described by a NTK. Though the filter may not be invariant under translation, and propose embedding leads to spatial invariance of the NTK, and filter.

### Fast Graph Neural Tangent Kernel via Kronecker Sketching

<https://arxiv.org/abs/2112.02446>

Sketching has become increasingly used in speeding up kernel regression, however it takes O(n^2 N^4) for GNTK, and this paper provides first algorithm to construct kernel matrix in o(n^2 N^3) running time.

### Steps Toward Deep Kernel Methods from Infinite Neural Networks

<https://www.arxiv.org/abs/1508.05133>

Devise stochastic kernels that encode the information of networks. 

### Fast Adaptation with Linearized Neural Networks

<https://www.arxiv.org/abs/2103.01439>

Using trained network's linearization, derive speed up on meta learning techniques, showing inductive bias of NTK at training, with novel computation of Fisher vector product to efficiently compute NTK inference.

### Constrained Policy Gradient Method for Safe and Fast Reinforcement Learning: a Neural Tangent Kernel Based Approach

<https://www.arxiv.org/abs/2107.09139>

Under lazy learning so that episodic policy change can be computed using policy gradient and NTK, the learning can be guided ensuring safety via augmenting episode batches with states where desired action probabilities are prescribed.

### Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning

<https://www.arxiv.org/abs/2203.09137>

First prove that MAML with over parameterized neural network is guaranteed to converge to global optima in linear rate, and show that MAML is equivalent to kernel regression with a class of kernels named MetaNTK. Then propose MetaNTK-NAS which usese MetaNTK to rank and select architectures.

### Generative Adversarial Method Based on Neural Tangent Kernels

<https://www.arxiv.org/abs/2204.04090>

Replace discriminator with NTK based GP prediction, and conduct experiments that this approach solve three problems in GAN training, failure on convergence, mode collapse, and vanishing gradient in small data setting.

# Empirical Study

### Finite versus infinite neural networks: an empirical study

<https://www.arxiv.org/abs/2007.15801>

Empirical study between correspondence between wide neural networks and kernel methods.

### Analyzing Finite Neural Networks: Can We Trust Neural Tangent Kernel Theory?

<https://www.arxiv.org/abs/2012.04477>

Empirically study measures that checks NTK theory.

### Neural Kernels Without Tangents

<https://www.arxiv.org/abs/2003.02237>

Define compositional kernel and compare the accuracy.

### Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel

<https://www.arxiv.org/abs/2010.15110>

Compute some measures in training dynamics, to check whether NTK theory works.

### Harnessing the power of infinitely wide deep nets on small-data tasks

<https://www.arxiv.org/abs/1910.01663>

Empirically study NTK's performance in small-data task. NTKs perform strongly on low-data tasks.

### What can linearized neural networks actually say about generalization?

<https://www.arxiv.org/abs/2106.06770>

Show empirically that linear approximations can indeed rank the learning complexity of certain tasks for neural networks, even in very different performances. Also discover that neural networks do not always perform better than kernel approximation, and that performance gap highly depends on architecture, dataset, training tasks, showing that networks overfit due to evolution of kernel.

### Demystifying the Neural Tangent Kernel from a Practical Perspective: Can it be trusted for Neural Architecture Search without training?

<https://www.arxiv.org/abs/2203.14577>

Show that due to non-linear characteristics of modern architecture, NTK-based metrics for estimating the performance in NAS is incapable. Instead, propose Label-Gradient Alignment, which is NTK-based metric that capture the non-linear advantage.

### The training response law explains how deep neural networks learn

<https://www.arxiv.org/abs/2204.07291>

Study the learning process with simple supervised learning example, and find that in the training reponse, simple law descrbing NTK. The response is power law like decay multiplied by a simple response kernel, and conduct a simple mean-field dynamical model with given law, explaining how the network learns.