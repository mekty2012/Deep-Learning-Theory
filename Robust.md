### On the regularized risk of distributionally robust learning over deep neural networks

<https://www.arxiv.org/abs/2109.06297>

Using tools from optimal transport theory, derive first order and second order approximations to the distributionally robust problem in terms of appropriate regularized risk minimization problems. 

### The mathematics of adversarial attacks in AI -- Why deep learning is unstable despite the existence of stable neural networks

<https://www.arxiv.org/abs/2109.06098>

Show the mathematical paradox, that any training procedure with a fixed architecture will yield neural networks that are either inaccurate or unstable. The key is that the stable and accurate neural networks must have variable dimensions depending on the input.

### Understanding the Logit Distributions of Adversarially-Trained Deep Neural Networks

<https://www.arxiv.org/abs/2108.12001>

Provide a theoretical justification for the finding that adversarial training shrinks two important characteristics of the logit distribution: the max logit values and the logit gaps are on average lower for AT models. 

### On the Effect of Low-Rank Weights on Adversarial Robustness of Neural Networks

<https://www.arxiv.org/abs/1901.10371>

Show that adversarial training tends to promote simultaneously low-rank and sparse structure. In the reverse direction, when the low rank structure is promoted by nclear norm regularization, neural networks show significantly improved robustness.

### Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks

<https://www.arxiv.org/abs/2110.03825>

Provide a theoretical analysis explaning on following observations, that 1) model parameters does not necessarily help adversarial robustness, 2) reducing capacity at the last stage of the network can actually improve adversarial robustness, and 3) under the same parameter budget, there exists an optimal architectural configuration for adversarial robustness.

### Can we have it all? On the Trade-off between Spatial and Adversarial Robustness of Neural Networks

<https://www.arxiv.org/abs/2002.11318>

Prove a quantitative trade-off between spatial and adversarial robustness in a simple statistical setting. 

### How Does a Neural Network's Architecture Impact Its Robustness to Noisy Labels?

<https://www.arxiv.org/abs/2012.12896>

Provide a formal framework connecting the robustness of a network to the alignments between its architecture and target functions. Hypothesize that a network is more robust to noisy labels if its architecture is more aligned with the target function than the noise.

### Adversarial Noises Are Linearly Separable for (Nearly) Random Neural Networks

<https://www.arxiv.org/abs/2206.04316>

Prove that the adversarial noises crafted by one-step gradient methods are linearly separable, for a two-layer network. The proof idea is to show that the label infromation can be efficiently propagated to the input while keeping the linear separability.

### On the (Non-)Robustness of Two-Layer Neural Networks in Different Learning Regimes

<https://www.arxiv.org/abs/2203.11864>

Considering the over-parameterized networks in high dimensions with quaratic targets and infinite samples, identify the tradeoff between approximation and robustness. Also show that linearized lazy training regime can worsen robustness due to improperly scaled random initialization.

### Robustness Implies Generalization via Data-Dependent Generalization Bounds

<https://www.arxiv.org/abs/2206.13497>

Proves that robustness implies generalization via data-dependent generalization bound, which improves previous bound in two ways. First, reduce dependence on the covering number and remove the dependence on the hypothesis space.

### A Universal Law of Robustness via Isoperimetry

<https://www.arxiv.org/abs/2105.12806>

Show that smooth interpolation requires d times parameters than mere interpolation, where d is the ambient data dimension, for any smoothly parametrized function class with polynomial size weights, and any covaraiate distribution verifying isoperimetry.

### Improved Regularization and Robustness for Fine-tuning in Neural Networks

<https://www.arxiv.org/abs/2111.04578>

Present a PAC-Bayes generalization bound that depends on the distance traveled in each layer during fine-tuning and the noise stability. 

### Why Robust Generalization in Deep Learning is Difficult: Perspective of Expressive Power

<https://www.arxiv.org/abs/2205.13863>

Even the robust training accuracy can be near zero, show that there exists a constant generalization gap unless the size of network is exponential in the data dimension. This holds for various architectures, as long as their VC dimension is at most polynomial in the number of paramters. Also establish an improved upper bound for the network capacity, depending on the intrinsic dimension.

### Adversarial Robustness is at Odds with Lazy Training

<https://www.arxiv.org/abs/2207.00411>

Show that the over-parameterized neural networks that generalize well due to lazy training, remain vulnerable to attacks with single gradient ascent steps.

### Deep Learning is Provably Robust to Symmetric Label Noise

<https://www.arxiv.org/abs/2210.15083>

Show that for multiclass classification, L1-consistent DNN classifiers trained with symmetric label noise achieve Bayes optimality asymptotically.

### On the uncertainty principle of neural networks

<https://www.arxiv.org/abs/2205.01493>

Show that the accuracy-robustness trade-off is an intrinsic property, and is closely related to the uncertainty principle. In specific, relate the loss function to the wave function in quantum mechanics, showing that both inputs and conjugate can't be resolved simultaneously.

### An Analysis of Robustness of Non-Lipschitz Networks

<https://www.arxiv.org/abs/2010.06154>

Prove that the adversarial attack that moves feature in low-dimensional subspace can be quite powerful. However allowing the network not to predict for unusal inputs, such adversaries can be overcome given well-separation assumption.

### Gradient Methods Provably Converge to Non-Robust Networks

<https://www.arxiv.org/abs/2202.04347>

Show that there exists robust parameter for 2-layer network, but gradient flow gives non-robust network which can be solved that KKT condition of the max-margin problem is non-robust.

### Adversarial Robustness is at Odds with Lazy Training

<https://www.arxiv.org/abs/2207.00411>

Show that the lazy-training of the neural network always lead to the vulnerable model, that can be attacked with single step of gradient ascent.

### On the Minimal Adversarial Perturbation for Deep Neural Networks with Provable Estimation Error

<https://www.arxiv.org/abs/2201.01235>

Propose two strategy to find the minimal adversarial perturbation, which allows to formulate an estimation error of approximate distance compared to theoretical one.

### Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels

<https://www.arxiv.org/abs/2302.01629>

Prove that for random features, the model is not robust for any degree of over-parameterization. In contrast, the NTK model meets the universal lower bound, and is robust as soon as the necessary condition of over-parameterization is satisfied.

### Adversarial Examples Exist in Two-Layer ReLU Networks for Low Dimensional Data Manifolds

<https://www.arxiv.org/abs/2303.00783>

For two-layer ReLU networks trained on the data in low dimensional linear subspace, show that gradient method lead to non-robust neural network having large gradient orthogonal to the data subspace.

### The Double-Edged Sword of Implicit Bias: Generalization vs. Robustness in ReLU Networks

<https://www.arxiv.org/abs/2303.01456>

Show that the gradient flow of ReLU NN is biased towards solutions that generalize well, but highly vulnerable to adversarial exmaples.

### How many dimensions are required to find an adversarial example?

<https://www.arxiv.org/abs/2303.14173>

Show that the adversarial success of standard PGD attacks on Lp norm constraint behaves like a monotonically increasing function epsilon * (dim(V) / dim(X))^(1/q) where V is the subspace of attack and X is input space. 

### Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels

<https://www.arxiv.org/abs/2302.01629>

Prove that random feature models are not robust at any degree of overparameterizations, but NTK models meets the universal lower bound and robust if necessarity on overparamterization is fulfilled.

### Sup-Norm Convergence of Deep Neural Network Estimator for Nonparameteric Regression by Adversarial Training

<https://www.arxiv.org/abs/2307.04042>

Show the sup-norm convergence of DNNs with adversarial training, for new adversatrial trainnig scheme and estiamtor achieves the optimal rates.

### On the Robustness of Bayesian Neural Networks to Adversarial Attacks

<https://www.arxiv.org/abs/2207.06154>

Analyze the overparameterized limit of BNNs and gardient based attacks, and show that the vulnerability is because data lies in lower-dimensional submanifold. Then prove that the expected gradient of the loss for posterior distribution vanishes even if each neural networks are weak.

### Adversarial Examples Exist in Two-Layer ReLU Networks for Low Dimensional Linear Subspaces

<https://arxiv.org/abs/2303.00783>

Consider the case where neural network is trained by data on low dimensional linear subspace, and show that stnadard gradient optimisations lead non-robust neural networks, that have large gradient to the direction orthogonal to data subspace. Also show that small initialisation scale or L2 regularsiation can make the trained neural network robust.

### The Double-Edged Sword of Implicit Bias: Generalization vs. Robustness in ReLU Networks

<https://arxiv.org/abs/2303.01456>

Show that the gradient flow of two-layer ReLU net is biased toward solution with good generalization, but highly vulnerable to adversarial examples, in case where data has clusters and correlation between clusters are small. This holds even in overparameterised cases, where the implicit bias solves the overfitting problem, but still leads to non-robust neural networks.