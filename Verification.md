### AI2: Safety and Robustness Certification of Neural Networks with Abstract Interpretation

<https://ieeexplore.ieee.org/document/8418593>

Use abstract interpretation in verifying the neural network, containing affine transformations.

### Robustness Verification for Transformers

<https://www.arxiv.org/abs/2002.06622>

By computing the global lower bound and global upper bound for each neurons, approximate the maximum adversarial accuracy by lower bound.

### Robustness Certification with Generative Models

<https://files.sri.inf.ethz.ch/website/papers/mirman2021pldi.pdf>

Using the abstract interpretation of union of lines and their approximation with boxes, prove deterministic and probabilistic properties of generative models.

### Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks

<https://www.arxiv.org/abs/1702.01135>

By recording the maximum and minimum value of activations, verifies the robustness with linear equation SMT solver.

### Fast and precise certification of transformers

<https://dl.acm.org/doi/10.1145/3453483.3454056>

Using Zonotope with multi-norm, define the abstract interpretation for Transformer network. Gives two rule of computing abstract interpretation, one is precise but slow, 
the other slightly over approximates but faster.

### Reachability Analysis of Deep Neural Networks with Provable Guarantees

<https://www.arxiv.org/abs/1805.02242>

Using the Lipscitz continuity property of DNN, computes the exact lower bound and upper bound of function values, and use it for safety verification problem.

### An Inductive Synthesis Framework for Verificable Reinforcement Learning

<https://www.arxiv.org/abs/1907.07273>

Approximates the RL system by piecewise linear function and finds safety invariant of approximations. If invariant was violated in RL system, use approximation instead.

### Scalable and Modular Robustness Analysis of Deep Neural Networks

<https://www.arxiv.org/abs/2108.11651>

Propose the network block summarization technique to capture the behaviors within a network block using a block summary, and leverage the summary to speed up the analysis process. By segmenting a network into blocks and conduct the analysis for each block, modularly analyzes neural networks.

### Proof Transfer for Neural Network Verification

<https://www.arxiv.org/abs/2109.00542>

Show that by generating proof templates that capture and generalize existing proofs, can speed up subsequent proofs. We create these templates from previous proofs on the same neural network.

### Provably Correct Training of Neural Network Controllers Using Reachability Analysis

<https://www.arxiv.org/abs/2102.10806>

First compute a finite state abstract model that captures the closed-loop behavior under all possible CPWA controllers, then identifies a family of CPWA functions satisfy the safety. Then augment the learning algorithm with a NN weight projection operator during training, enforcing the NN to represent a CPWA function of such family.

### Global Optimization of Objective Functions Represented by ReLU Networks

<https://www.arxiv.org/abs/2010.03258>

Extend existing verifiers to perform optimization and find the most extreme failure in a given input region and the minimum input perturbation required to cause a failure.

### Safety Verification and Robustness Analysis of Neural Networks via Quadratic Constraints and Semidefinite Programming

<https://www.arxiv.org/abs/1903.01287>

Propose a semidefinite programming framework for feed-forward neural networks with general activation functions. Abstract various properties of activations functions with the quadratic constraints, then analyze the safety property via the S-procedure.

### Shared Certificates for Neural Network Verification

<https://www.arxiv.org/abs/2109.00542>

Introduce a new method for general concept of shared certificates, enabling proof effort reuse across multiple inputs and driving down overall verification costs.

### Verifying Quantized Neural Networks using SMT-Based Model Checking

<https://www.arxiv.org/abs/2106.05997>

Develop a novel symbolic verification framework using SMC and SMT to check vulnerabilities in ANNs, propose several ANN-related optimizations for SMC, including invariant inference via interval analysis, slicing, expression simplifications, and discretization of activation functions.

### CC-Cert: A Probabilistic Approach to Certify General Robustness of Neural Networks

<https://www.arxiv.org/abs/2109.10696>

Propose a new universal probabilistic certification approach based on Chernoff-Cramer bounds that can be used in general attack settings.

### A Sequential Framework TOwards an Exact SDP Verification of Neural Networks

<https://www.arxiv.org/abs/2010.08603>

Adress SDP's problem that is prone to a large relaxation gap by developing a sequential framework to shring the gap to zero by adding non-convex cuts to the optimization problem via disjunctive programming.

### Continuous Safety Verification of Neural Networks

<https://www.arxiv.org/abs/2010.05689>

Develop several sufficient conditions that only require formally analyzing a small part of the DNN in the new problem, by reusing state abstractions, network abstractions, and Lipscitz constants.

### Local Repair of Neural Networks Using Optimization

<https://www.arxiv.org/abs/2109.14041>

Define the NN repair problem as a Mixed Integer Quadratic Program to adjust the weights of a single layer subject to the given predicates while minimizing the original loss function over the original training domain.

### Improved Branch and Bound for Neural Network Verification via Lagrangian Decomposition

<https://www.arxiv.org/abs/2104.06178>

Propose novel bounding algorithm based on Lagrangian decomposition which restricts the optimization to a subspace of the dual domain, resulting in accelerated convergence with parallel implementation, with activation based branching strategy.

### FedlPR: Ownership Verification for Federated Deep Neural Network Models

<https://www.arxiv.org/abs/2109.13236>

Add ownership verification scheme that allows signatures to be embedded and verified to claim legitimate intellectural property rights, when models are illegally copied, re-distributed or misused. 

### Permutation Invariance of Deep Neural Networks with ReLUs

<https://www.arxiv.org/abs/2110.09578>

Proposes a sound, abstraction-based technique to establish permutation invariance in DNNs. The technique computes an over-approximation of the reachable state, and an under-approximation of the safe states, and propagates this information both forward and backward. 

### Minimal Multi-Layer Modifications of Deep Neural Networks

<https://www.arxiv.org/abs/2110.09929>

Computes a modification to the netowrk's weights that corrects its behavior, and attempts to minimize this change via a sequence of calls to a backend, black-box DNN verification engine. Splitts the network into sub-networks, and apply a single-layer repairing technique to each component, allowing repair of the network by simulateneously modifying multiple layers.

### Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers

<https://www.arxiv.org/abs/2011.13824>

Propose to use the backward mode linear relaxation based perturbation analysis to replace linear programming during the brandch-and-bound process, that can be efficiently implemented in GPUs. LiPRA can produce much weaker bounds, so apply a fast gradient based bound tightening procedure combined with batch splits.

### Static analysis of ReLU neural networks with tropical polyhedra

<https://www.arxiv.org/abs/2108.00893>

Abstracts ReLU feedforward neural networks, and show that tropical polyhedra can efficiently abstract ReLU activation function, while being able to control the loss of precision due to linear computations. Show how the connection between ReLU networks and tropical rational functions can provide approaches for range analysis of ReLU neural networks.

### Exploiting Verified Neural Networks via Floating Point Numerical Error

<https://www.arxiv.org/abs/2003.03021>

Show that the negligence of floating point error leads to unsound verification that can be systematically exploited in practice. Present a method that efficiently searches inputs as witnesses for the incorrectness of robustness claims made by a complete verifier.

### Verifying Low-dimensional Input Neural Networks via Input Quantization

<https://www.arxiv.org/abs/2108.07961>

Prepend input quantization layer to the network, which allows efficient verficiation via input state enumeration. 

### Reduced Products of Abstract Domains for Fairness Certification of Neural Networks

<https://link.springer.com/chapter/10.1007/978-3-030-88806-0_15>

By combining a sound forward pre-analysis and an exact backward analysis, leverages the polyhedra abstract domain to provide definite fairness guarantees when possible, and to otherwise quantify and describe the biased input space regions.

### PRIMA: Precise and General Neural Network Certification via Multi-Neuron Convex Relaxations

<https://www.arxiv.org/abs/2103.03638>

Using convex hull approximation algorithms from computational geometry, create precise and general verification methods with polynomial complexity.

### Interval Universal Approximation for Neural Networks

<https://www.arxiv.org/abs/2007.06093>

Shows that neural networks not only can approximate any continuous function, but can find neural networks with arbitrarily close interval bound. Constructing such neural network takes Delta 2-intermediate program, which is strictly harder than NPc problems.

### A Dual Number Abstraction for Static Analysis of Clarke Jacobians

<https://popl22.sigplan.org/details/POPL-2022-popl-research-papers/56/A-Dual-Number-Abstraction-for-Static-Analysis-of-Clarke-Jacobians>

Design new abstract interpretation that bounds Clarke Jacobian, over-approximate the gradient.

### epsilon-weakened Robustness of Deep Neural Networks

<https://www.arxiv.org/abs/2110.15764>

Define epsilon-weakened robustness which allows proportion of adversarial examples, and prove that decision problem is PP-complete, and finally devise an algorithm to find the maximum epsilon-weakened robustness radius.

### Traversing the Local Polytopes of ReLU Neural Networks: A Unified Approach for Network Verification

<https://www.arxiv.org/abs/2111.08922>

Design a traversing algorithm based on adjacency of local polytopes, which can be adapted to verify network properties related to robustness and interpretability. 

### Fast BATLLNN: Fast Box Analysis of Two-Level Lattice Neural Networks

<https://www.arxiv.org/abs/2111.09293>

Design a verification tool on two-level lattice neural networks, that determines whether convex polytope in input's output always lies within a specified hyper-rectangle. Using the decoupled nature of box-like output contraint, improve verification performance.

### AMITE: A Novel Polynomial Expansion for Analyzing Neural Network Nonlinearities

<https://www.arxiv.org/abs/2007.06226>

Develop an analytically modieifed integral transform expansion, a novel expansion via integral transforms modified using derived criteria for convergence. This can provide six desired properties like exact formulas and exact expansion. As a result, multivariate polynomial form can be efficiently extracted to facilitate equivalence testing and a variety of NN architectures having 3~7 layers are bounded using Taylor models.

### QNNVerifier: A Tool for Verifying Neural Networks using SMT-Based Model Checking

<https://www.arxiv.org/abs/2111.13110>

Translate the implementation of neural networks to a decidable fragment of first-order logic based SMT, where floating-point operations are represented by direct implementation given a hardware determined precision. 

### ArchRepair: Block-Level Architecture-Oriented Repairing for Deep Neural Networks

<https://www.arxiv.org/abs/2111.13330>

Repair DNNs by jointly optimizing the architecture and weights. Propose adversarial-aware spectrum analysis for vulnerable block localization, which enables more accurate candidate localization. Then do architecture-oriented search-based repairing.

### SoK: Certified Robustness for Deep Neural Networks

<https://www.arxiv.org/abs/2009.04131>

Systemize the certifiably robust approaches and related practical and theoretical implications, provide the first comprehensive benchmark on existing robust verification and training approaches. 

### The Fundamental Limits of Interval Arithmetic for Neural Network

<https://www.arxiv.org/abs/2112.05235>

Show that any neural network classifying just three points, there is a valid specifiation that interval analysis can not prove. In one-hidden-layer network, show that there is O(x^-1) points at robust radius x, can not be proven through interval analysis. 

### Geometric Path Enumeration for Equivalence Verification of Neural Networks

<https://www.arxiv.org/abs/2112.06582>

Tries to formally verify equivalence of two NNs, for showing correctness of compressed version using Path Enumeration algorithm.

### An Abstraction-Refinement Approach for Verifying Convolution Neural Networks

<https://www.arxiv.org/abs/2201.01978>

Resolve scalability issue for CNN verification, uses abstraction refinement technique simplifying the problem through removal of convolution connection, generating over-approximation of original problem, and restore connections if problem is too abstract.

### Verified Probabilistic Policies for Deep Reinforcement Learning

<https://www.arxiv.org/abs/2201.03698>

Propose an abstraction approach based on interval Markov decision process, yields probabilistic guarantees on a policy's execution and techniques to solve models using abstract interpretation, integer linear programming, entropy based refinement, probabilistic model checking.

### DeepGalaxy: Testing Neural Network Verifiers via Two-Dimensional Input Space Exploration

<https://www.arxiv.org/abs/2201.08087>

Propose an automated approach based on differential testing, tests neural network verifiers based on mutation rules and heuristic strategies to select test cases.

### DeepSplit: Scalable Verification of Deep Neural Networks via Operator Splitting

<https://www.arxiv.org/abs/2106.09117>

Propose a novel operator splitting method that can directly solve a convex relaxation of the optimization problem to high accuracy, by splitting it into smaller sub-problems that often have analytical solutions.

### Individual Fairness Guarantees for Neural Networks

<https://www.arxiv.org/abs/2205.05763>

Using Mixed-Integer Linear Programming to overapproximate the individual fairness verification.

### Abstraction and Refinement: Towards Scalable and Exact Verification of Neural Networks

<https://www.arxiv.org/abs/2207.00759>

Propose a novel abstraction to break down the size of DNNs by over-approximation, and counterexample-guided refinement that eliminate spurious counterexamples.

### Fundamental Limits in Formal Verification of Message-Passing Neural Networks

<https://www.arxiv.org/abs/2206.05070>

Show that the output reachability of graph-classifier MPNN over graphs with unbounded size, with sufficiently expressive node labels cannot be verified formally. However, such verification is possible when a limit on the degree is given.

### Verifying and Interpreting Neural Networks using Finite Automata

<https://www.arxiv.org/abs/2211.01022>

Show that the input-output behaviour of a DNN can be captured precisely by a weak Buchi automaton with exponential size, and these can address verification tasks like adversarial robustness and minimum sufficient reasons.

### Probabilistic Verification of ReLU Neural Networks via Characteristic Functions

<https://www.arxiv.org/abs/2212.01544>

Interpret the DNN as discrete dynamical system, and use characteristic function to propagate the distribution of the input data. Using the inverse Fourier transform, obtain the cumulative distribution function of the ouptut set, that can be used to check if the network is performing as expected.

### Fairify: Fairness Verification of Neural Networks

<https://www.arxiv.org/abs/2212.06140>

Using the SMT based approach, verify individual fairness that two similar individual get similar treatment independent to protected attribute, using the idea that many neurons remain inactive when a smaller part of input is considered.