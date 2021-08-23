## NTK theory

### Deep neural networks as Gaussian processes
<www.arxiv.org/abs/1711.00165>
In the limit of infinite network width, the neural network becomes a Gaussian process.

### Neural tangent kernel: convergence and generalization in neural networks
<arxiv.org/abs/1806.07572>
Empirical kernel converges to deterministic kernel, and remains constant in 

### Wide neural networks of any depth evolve as linear models under gradient descent
<arxiv.org/abs/1902.06720>
Infinite neural network follows linear dynamics, and can be solved by solving linear ODE.

### Finite versus infinite neural networks: an empirical study
<arxiv.org/abs/2007.15801>
Empirical study between correspondence between wide neural networks and kernel methods.

### Bayesian deep ensembles via the Neural tangent kernel
<arxiv.org/abs/2007.05864>
Using the NTK, add bias to initialization derive bayesian interpretation for deep ensembles.
, 
### Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel
<arxiv.org/abs/2010.15110>
Compute some measures in training dynamics, to check whether NTK theory works.

### Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes
<arxiv.org/abs/1910.12478>
Compute convergence to coordinate distribution for simple network. Can prove NNGP correspondence for universal architectures.

### Tensor Programs II: Neural Tangent Kernel for Any Architecture
<arxiv.org/abs/2006.14548>
Compute convergence to coordinate distribution for BP-like network, can use transpose. Can prove NTK convergence in initialization.

### Tensor Programs III: Neural Matrix Laws
<arxiv.org/abs/2009.10685>
Compute convergence to coordinate distribution, can use transpose. Can prove asymptotic freeness of matrices.

### Tensor Programs IIb: Architectural Universality of Neural Tangent Kernel Training Dynamics
<arxiv.org/abs/2105.03703>
Prove NTK convergence in training. Derive back propagation as a vector in program.

### Feature Learning in Infinite-Width Neural Networks
<arxiv.org/abs/2011.14522>
Define feature learning in terms of asymptotic size of output, design new parametrization.

### Neural Kernels Without Tangents
<arxiv.org/abs/2003.02237>
Define compositional kernel and compare the accuracy.

### On Infinite-Width Hypernetworks
<arxiv.org/abs/2003.12193>
Studies GP, NTK behavior of hypernetwork, which computes weights for a primiary network.

### Mean-field Behaviour of NTK
<arxiv.org/abs/1905.13654>
Bridge the gap between NTK theory and EOC initialization.

### Analyzing Finite Neural Networks: Can We Trust Neural Tangent Kernel Theory?
<arxiv.org/abs/2012.04477>
Empirically study measures that checks NTK theory.

### On Random Kernels of Residual Architectures
<arxiv.org/abs/2001.10460>
Study finite width and depth corrections for the NTK of ResNets and DenseNets. 
Finite size residual architecture are initialized much closer to the kernel regime than vanilla.

### The recurrent neural tangent kernel
<arxiv.org/abs/2006.10246>
Study NTK for recurrent neural networks. 

### Infinite Attention: NNGP and NTK for Deep Attention Networks
<arxiv.org/abs/2006.10540>
Study NTK for attention layers, and propose modifiationc of the attention mechanism.

### Scaling limits of wide neural networks with weight sharing: Gaussian process behavior, gradient independence, and neural tangent kernel derivation
<arxiv.org/abs/1902.04760>
The very beginning of tensor program. 

### Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes
<arxiv.org/abs/1810.05148>
Compute CNN both with and without pooling layers' equivalence to GP. Also introduce Monte Carlo method to estimate the GP to corresponding NN architecture.
Without pooling, weight sharing has no effect in GP, iplying that translation equivariance has no role in bayesian infinite limit. 

### Harnessing the power of infinitely wide deep nets on small-data tasks
<arxiv.org/abs/1910.01663>
Empirically study NTK's performance in small-data task. NTKs perform strongly on low-data tasks.

### Finite depth and width corrections to the neural tangent kernel
<arxiv.org/abs/1909.05989>
Prove teh precise scaling for the mean and variance of the NTK. When both depth and width tends to infinity, NTK is no more deterministic with non-trivial evolution.

### Dynamics of deep neural networks and neural tangent hierarchy
<arxiv.org/abs/1909.08156>
Study NTK dynamics for finite width deep FCNNs. Derive an infinite hierarchy of ODEs, the neural tangent hierarchy which captures the gradient descent dynamic.
The truncated hierarchy can approximate the dynamic of the NTK up to arbitrary precision.

### On the neural tangent kernel of deep networks with orthogonal initialization
<arxiv.org/abs/2004.05867?
Study dynamics of ultra-wide networks including FCN, CNN with orthogonal initialization via NTK. 
Prove that Gaussian weight and orthogonal weight's NTK are equal in infinite width, and both stays constant. 
It suggests that orthogonal initialization does not speed up training. 

### Gaussian Process Behavious in Wide Deep Neural Networks
<arxiv.org/1804.11271>
From NNGP correspondence, empirically evaluates convergence rate and compare with Bayesian deep networks.

### On Exact Computation with an Infinitely Wide Neural Net
<arxiv.org/abs/1904.11955>
First efficient exact algorithm for computing the extension of NTK to CNN, as well as an efficient GPU implementation. 

### Finding sparse trainable neural networks through Neural Tangent Transfer
<arxiv.org/abs/2006.08228>
Introduce Neural Tangent Transfer, a method that finds trainable sparse networks in a label-free manner, that whose training dynamics computed by NTK is similar to dense ones.

### Neural tangent kernels, transportation mappings, and universal approximation
<arxiv.org/abs/1910.06956>
A generic scheme to approximate functions with the NTK by sampling and the construction of transport mappings via Fourier transforms.

### Beyond Linearization: On Quadratic and Higher-Order Approximation of Wide Neural Networks
<arxiv.org/abs/1910.01619>
Investigate the training of over-parametrized NNs that are beyoung the NTK regime, yet still governed by the Taylor expansion.

### Towards Understanding Hierarchical Learning: Benefits of Neural Representations
<arxiv.org/abs/2006.13436>
Using random wide two-layer untrainable networks as a representation function, if the trainable network is the quadratic Taylor model of a wide two-layer network,
neural representation can achieve improved sample complexities. But this does not increase in NTK.
