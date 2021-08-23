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
