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
