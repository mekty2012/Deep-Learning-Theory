## NTK theory

### Deep neural networks as Gaussian processes

<https://www.arxiv.org/abs/1711.00165>

In the limit of infinite network width, the neural network becomes a Gaussian process.

### Neural tangent kernel: convergence and generalization in neural networks

<https://www.arxiv.org/abs/1806.07572>

Empirical kernel converges to deterministic kernel, and remains constant in 

### Wide neural networks of any depth evolve as linear models under gradient descent

<https://www.arxiv.org/abs/1902.06720>

Infinite neural network follows linear dynamics, and can be solved by solving linear ODE.

### Finite versus infinite neural networks: an empirical study

<https://www.arxiv.org/abs/2007.15801>

Empirical study between correspondence between wide neural networks and kernel methods.

### Bayesian deep ensembles via the Neural tangent kernel

<https://www.arxiv.org/abs/2007.05864>

Using the NTK, add bias to initialization derive bayesian interpretation for deep ensembles.

### Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel

<https://www.arxiv.org/abs/2010.15110>

Compute some measures in training dynamics, to check whether NTK theory works.

### Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes

<https://www.arxiv.org/abs/1910.12478>

Compute convergence to coordinate distribution for simple network. Can prove NNGP correspondence for universal architectures.

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

### Neural Kernels Without Tangents

<https://www.arxiv.org/abs/2003.02237>

Define compositional kernel and compare the accuracy.

### On Infinite-Width Hypernetworks

<https://www.arxiv.org/abs/2003.12193>

Studies GP, NTK behavior of hypernetwork, which computes weights for a primiary network.

### Mean-field Behaviour of NTK

<https://www.arxiv.org/abs/1905.13654>

Bridge the gap between NTK theory and EOC initialization.

### Analyzing Finite Neural Networks: Can We Trust Neural Tangent Kernel Theory?

<https://www.arxiv.org/abs/2012.04477>

Empirically study measures that checks NTK theory.

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

### Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes

<https://www.arxiv.org/abs/1810.05148>

Compute CNN both with and without pooling layers' equivalence to GP. Also introduce Monte Carlo method to estimate the GP to corresponding NN architecture.
Without pooling, weight sharing has no effect in GP, iplying that translation equivariance has no role in bayesian infinite limit. 

### Harnessing the power of infinitely wide deep nets on small-data tasks

<https://www.arxiv.org/abs/1910.01663>

Empirically study NTK's performance in small-data task. NTKs perform strongly on low-data tasks.

### Finite depth and width corrections to the neural tangent kernel

<https://www.arxiv.org/abs/1909.05989>

Prove the precise scaling for the mean and variance of the NTK. When both depth and width tends to infinity, NTK is no more deterministic with non-trivial evolution.

### Dynamics of deep neural networks and neural tangent hierarchy

<https://www.arxiv.org/abs/1909.08156>

Study NTK dynamics for finite width deep FCNNs. Derive an infinite hierarchy of ODEs, the neural tangent hierarchy which captures the gradient descent dynamic.
The truncated hierarchy can approximate the dynamic of the NTK up to arbitrary precision.

### On the neural tangent kernel of deep networks with orthogonal initialization

<https://www.arxiv.org/abs/2004.05867?

Study dynamics of ultra-wide networks including FCN, CNN with orthogonal initialization via NTK. 
Prove that Gaussian weight and orthogonal weight's NTK are equal in infinite width, and both stays constant. 
It suggests that orthogonal initialization does not speed up training. 

### Gaussian Process Behavious in Wide Deep Neural Networks

<https://www.arxiv.org/1804.11271>

From NNGP correspondence, empirically evaluates convergence rate and compare with Bayesian deep networks.

### On Exact Computation with an Infinitely Wide Neural Net

<https://www.arxiv.org/abs/1904.11955>

First efficient exact algorithm for computing the extension of NTK to CNN, as well as an efficient GPU implementation. 

### Finding sparse trainable neural networks through Neural Tangent Transfer

<https://www.arxiv.org/abs/2006.08228>

Introduce Neural Tangent Transfer, a method that finds trainable sparse networks in a label-free manner, that whose training dynamics computed by NTK is similar to dense ones.

### Neural tangent kernels, transportation mappings, and universal approximation

<https://www.arxiv.org/abs/1910.06956>

A generic scheme to approximate functions with the NTK by sampling and the construction of transport mappings via Fourier transforms.

### Beyond Linearization: On Quadratic and Higher-Order Approximation of Wide Neural Networks

<https://www.arxiv.org/abs/1910.01619>

Investigate the training of over-parametrized NNs that are beyoung the NTK regime, yet still governed by the Taylor expansion.

### Towards Understanding Hierarchical Learning: Benefits of Neural Representations

<https://www.arxiv.org/abs/2006.13436>

Using random wide two-layer untrainable networks as a representation function, if the trainable network is the quadratic Taylor model of a wide two-layer network,
neural representation can achieve improved sample complexities. But this does not increase in NTK.

### On the expected behaviour of noise regularised deep neural networks as Gaussian processes

<https://www.arxiv.org/abs/1910.05563>

Consider impact of noise regularizations on NNGPs, and relate their behaviour to signal propagation theory.

### Neural Tangents: Fast and Easy Infinite Neural Networks in Python

<https://www.arxiv.org/abs/1912.02803>

High level API for specifying complex and hierarchical neural network architectures. 

### Infinitely Wide Graph Convolutional Networks: Semi-supervised Learning via Gaussian Processes

<https://www.arxiv.org/abs/2002.12168>

Inverstigate NNGP in GCNN, and propose a GP regression model with GCN, for graph-based semi-supervised learning.

### On the infinite width limit of neural networks with a standard parameterization

<https://www.arxiv.org/abs/2001.07301>

Propose an imporved extrapolation of the standard parameterization that yields a well-defined NTK. 

### Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Models

<https://www.arxiv.org/abs/1905.13192>

Presents a new class of graph kernel, Graph Neural Tangent Kernels which corerespond to infinitely wide multi-layer GNNs trained by gradient descent.
GNTKs provably learn a class of smooth functions on graphs.

### Regularization Matters: Generalization and Optimization of Neural Nets v.s. their Induced Kernel

<https://www.arxiv.org/abs/1810.05369>

Sample efficiency depend on the presence of the regularizer, that regularized NN requires O(d) samples but the NTK requires O(d^2) samples. 

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

### Limitations of Lazy Training of Two-layers Neural Networks

<https://www.arxiv.org/abs/1906.08899>

In two-layers neural networks with quadratic activations, (RF) train only last layer (NT) linearized dynamics (NN) full training has unbounded gap of prediction risk.

### Wide Neural Networks with Bottlenecks are Deep Gaussian Processes

<https://www.arxiv.org/abs/2001.00921>

In infinite network with bottleneck, which is some finite width hidden layers, the result is NNGP, which is composition of limiting GPs.

### Deep Convolutional Networks as shallow Gaussian Processes

<https://www.arxiv.org/abs/1808.05587>

Show that the output of a CNN with an appropriate prior is a Gaussian Process in the infinite channel limit, and can be computed efficiently similar to a single forward pass through the original CNN with single filter per layer.

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

### Why Do Deep Residual Networks Generalize Better than Deep Feedforward Networks? -- A Neural Tangent Kernel Perspective

<https://www.arxiv.org/abs/2002.06262>

Show that training ResNets can be viewed as learning reproducing kernel functions with some kernel function. THen compare the kernel of two networks, and show that the class of functions induced by FFNets is asymptotically not learnable, which does not happens in ResNets.

### Meta-Learning with Neural Tangent Kernels

<https://www.arxiv.org/abs/2102.03909>

Generalize MAML to function space, eliminating need of sub-optimal iterative inner-loop adaption by replacing the adaption with a fast-adaptive regularizer in the RKHS and solving the adaption analytically based on the NTK theory.

### Learning with Neural Tangent Kernels in Near Input Sparsity Time

<https://www.arxiv.org/abs/2104.00415>

Accelerate kenrel machines with NTK, by mapping the input data to a randomized low-dimensional feature space so that the inner product of the transformed data approximates the NTK evaluation, based on polynomial expansion of arc-cosine kernels.

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

### FL-NTK: A Neural Tangent Kernel-based Framework for Federated Learning Convergence Analysis

<https://arxiv.org/abs/2105.05001>

Presents a new class of convergence analysis for federated learning, which corresponds to overparameterized ReLU NNs trained by gradient descent in FL. Theoretically FL-NTK converges to a global optimal solution at a linear rate, and also achieve good generalizations.

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

### Scale Mixtures of Neural Network Gaussian Processes

<https://arxiv.org/abs/2107.01408>

Show that simply introducing a scale prior on the last-layer parameters can turn infinitely wide neural networks of any architecture into a richer class of stochastic process, like heavy-tailed stochastic processes.

### alpha-Stable convergence of heavy-tailed infinitely-wide neural networks

<https://arxiv.org/abs/2106.11064>

Assuming that the weights of an MLP are initialized with i.i.d. samples from either a light-tailed or heavy-tailed distribution in the domain of attraction of a symmetric alpha-stable distribution for alpha in (0, 2]. Show that the vector of pre-activation values at all nodes of a given hidden layer converges in the limit, to a vecctor of i.i.d. random variables with symmetric alpha-stable distributions.

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
