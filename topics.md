### Topology

Topology is a branch of mathematics that consider the space's property, via various tools including the Manifold theory and algebraic topology.
Algebraic topology is a branch of mathematics that analyze the geometry of space using tools like homology and homotopy. 
One of tool mainly used here is __Persistent Homology__, which records the change of space for varying parameter, and use this information to analyze the complexity of networks.
Manifold is a generalization of curve or curved surfaces, which is well used to represent some spaces in Euclidean space, where neural networks sometimes show that they learn this structure.

### Approximation

One of the big theorem in Deep Learning Theory is __Universal Approximation Theorem__, which states that any _good_ functions can be approximated by neural network with enough width. 
This result has been generalized to other functions and other neural networks, or different metric and problems.

### Bayesian

Uncertainty has been large topic in Deep Learning, which allows network to be less confident on some data.
This suggested doing Bayesian inference on neural network, while using the structure of neural network.
The theoretical results related to Bayesian neural networks are contained in this topics, even if the result itself is related to other topics.

### Complexity Theory

Computational Complexity is a branch of theoretical computer science, that analyze how many resources like time or space are required to solve the given problem. In the context of neural network, training or reachability has been analyzed.

### Differential Equation

Several differential equations including ODE, PDE, SDE, has been used in deep neural networks and their theory, like Neural ODE, or viewing the gradient flow as differential equation.

### Functional Analysis

Functional analysis is a branch of mathematics that considers functions space with norms.
Some of results focus on this norm of the neural network, analyzing the expressivity of bounded-norm NNs, and what functions do they include.

### GNN

Graph neural network is one of major topics in Deep Learning, which allows the input to be organized as graph.
Using the tools from graph theories, some results specific to graph neural networks has been analyzed.
The theoretical results related to Graph neural networks are contained in this topics, even if the result itself is related to other topics.

### Infinite Network

One of the simplifying assumption in DL theory is the limit sending the architecture parameter (width, depth, etc) to infinite.
Such an assumption has given Neural Network Gaussian Process Correspondence and Neural Tangent Kernel, which is a strong tool to analyze the training of neural network.
The theoretical results related to Infinite neural networks are contained in this topics, even if the result itself is related to other topics.

### Invariance

When the target function is invariant or equivariant for some action, giving similar structure to neural network increases the performance.
The theoretical results related to (In-/Equi-)variant neural networks are contained in this topics, even if the result itself is related to other topics.

### Learning Theory

Learning Theory focus on the generalization bound of various machine learning algorithms, trained on finite sample.
Because neural network contains so many parameters, traditional generalization bounds do not directly applied to neural networks.
Still there are continuous works that derives generalization error of neural network, for various data assumptions and architectures.

### Mean Field Theory

Mean Field theory is a branch of physics, that considers high dimensional stochastic models, via the approximation by averaging.
Since neural network has high dimensional parameters with random initialization, MFT has been used to analyze their properties.

### Optimization and Loss

It is different problem asking whether there exists neural network approximating solution, or we can find such neural network via training. 
Using various techniques in optimization theory, the convergence of training algorithms has been analyzed.
Also the loss landscape itself has interesting properties related to other topics, and such papers are also included here.

### Other networks / Others

The papers that do not matches other topics are contained here, where __Other Networks__ contains papers related to specific neural networks other than MLP or GNN.
If there are too many papers here, I check them to either create new category, or re-classify some papers.

### Physics

Tools from physics like effective theories are presented here, which allows to analyze neural networks with some loss of precision, but still gives good approximation of behaviors.

### Random Matrix Theory

Random Matrix Theory is a subject of mathematics, that studies the behaviors of large random matrices, especially their eigenvalues.
Since neural network itself has large nature while the data point also being large, some of approaches also applies to neural network.

### Robust

Robustness of neural network has been big problem for safety of deep learning, which considers the corruption on the input data, that can change the output of network largely.

### Verification

Proving that the instance of neural network has the property we have is complex, but using the tools from software verification like abstract interpretation, we can prove such properties automatically.