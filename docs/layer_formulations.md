## Background: Lipschitz Continuity
A function $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ is Lipschitz continuous if there exists a constant $L \geq 0$ such that:

$$|f(x) - f(y)| \leq L|x - y|$$
For all $x, y$ in the domain. For a linear layer $f(x) = Wx + b$, the Lipschitz constant is precisely the operator norm of $W$:
$$L = |W|_{op} = \sup_{x \neq 0} \frac{|Wx|}{|x|}$$

---

## 1. `StableSoftplusLipschitzLinear`
#### Mathematical Derivation
This approach uses a learnable parameter $c$ to control the Lipschitz constant:

$$\begin{align} c_{init} &= \log\max_i\left(\sum_j e^{|W_{ij}|}\right) L \\
&= \text{softplus}(c) = \log(1 + e^c) \ \text{scale}_i \\
&= \min\left(1, \frac{L}{\sum_j|W_{ij}|}\right) \ \hat{W}_{ij} \\
&= W_{ij} \cdot \text{scale}_i \\
f(x) &= \hat{W}x + b \end{align}$$

#### Pros and Cons
**Pros:**
- Differentiable through the softplus function
- Row-wise scaling preserves directional information
- Initialization based on log-sum-exp approximates the L∞ norm
**Cons:**
- Uses L1 row norm which is looser than spectral norm
- Numerical instability in softplus for large $c$ values
- May unnecessarily restrict expressivity

---

## 2. `GeometricMeanLipschitzLinear`
#### Mathematical Derivation
Similar to the previous approach but with a different numerical formulation:

$$\begin{align} c_{init} &= \log\left(\max_i \sum_j |W_{ij}|\right) \ L \\
&= e^{c + \log(1 + e^{-c})} \ \text{scale}_i \\
&= \min\left(1, \frac{L}{\sum_j|W_{ij}|}\right) \ \hat{W}_{ij} \\
&= W_{ij} \cdot \text{scale}_i \ \\
f(x) &= \hat{W}x + b \end{align}$$

#### Pros and Cons
**Pros:**
- Improved numerical stability through log-domain computation
- Geometric mean approach balances different norms
- Fully differentiable parameterization
**Cons:**
- Still uses row-wise L1 norm which is a loose upper bound
- Slower convergence compared to direct methods
- Scaling approach can cause rank collapse

---

## 3. `SpectralNormalizedLinear`
#### Mathematical Derivation
Directly normalizes the weight matrix by its spectral norm:

$$\begin{align} \sigma_{\max}(W) &= \text{largest singular value of }W \hat{W} \\
&= \frac{W}{\sigma_{\max}(W)} \ \\
f(x) &= \hat{W}x + b \end{align}$$

#### Pros and Cons
**Pros:**
- Uses exact spectral norm (tightest bound)
- Guaranteed Lipschitz constant of 1.0
- Efficient implementation using power iteration
- Preserves rank and directional information better
**Cons:**
- Less flexible than learnable approaches
- Power iteration adds computational overhead
- Harder to set specific Lipschitz values other than 1.0
- Autograd through SVD can be numerically unstable

---

## 4. `OrthogonalLipschitzLinear`
#### Mathematical Derivation
Parameterizes the weight matrix as a product of Householder reflections:

$$\begin{align} H_i &= I - 2v_iv_i^T/|v_i|^2 \quad \text{(Householder reflection)} \\ W &= \prod_{i=1}^{n} H_i \quad \text{(if expansive, padded with zeros)} \\
f(x) &= Wx + b \end{align}$$

#### Pros and Cons
**Pros:**
- Exact Lipschitz constant of 1.0 by construction
- Preserves norm exactly (isometric transformation)
- Efficient parameterization through reflectors
- Excellent gradient flow properties
- Numerically stable updates
**Cons:**
- Limited expressivity (only orthogonal transformations)
- Restricted to specific dimensions
- Sequential application limits parallelization
- Memory usage for householder vectors

---

## 5. `JacobianNormLipschitzLinear` (Commented Out)
#### Mathematical Derivation
Estimates the Lipschitz constant through Jacobian-vector products:

$$\begin{align} \hat{L} &= \max_{v \in V} \frac{|Jv|}{|v|} \quad \text{where } V = {v_1, \ldots, v_k} \text{ is a set of random unit vectors} \\
J &= W \text{ for a linear layer} \end{align}$$

#### Pros and Cons
**Pros:**
- Applicable to both linear and non-linear functions
- Adaptable to input distribution
- Does not require explicit matrix computation
- Usable for implicit layers
**Cons:**
- Stochastic estimation introduces variance
- Computational overhead from multiple samples
- For linear layers, more expensive than direct computation
- Suboptimal for pure linear transforms

---

## Theoretical Comparison
#### VC-Bound Effects

Tighter Lipschitz bounds generally reduce the VC-dimension:

$$\text{VC-dim}(f) \leq O\left(\frac{n L^2 \log(n L^2)}{\epsilon^2}\right)$$

Where $n$ is the number of parameters and $L$ is the Lipschitz constant.

#### Generalization Guarantees
- **Orthogonal/Spectral methods**: Strongest theoretical guarantees
- **Geometric Mean/Softplus methods**: Intermediate guarantees
- **Jacobian estimation**: Depends on sampling quality

#### Memory Efficiency
From most to least efficient:
1. `SpectralNormalizedLinear` (constant overhead)
2. `StableSoftplusLipschitzLinear`/`GeometricMeanLipschitzLinear` (scalar overhead)
3. `OrthogonalLipschitzLinear` (O(min(`in_dim`, `out_dim`)) overhead)
4. `JacobianNormLipschitzLinear` (O(`batch_size` $\times$ `in_dim`) overhead during estimation)

#### Computational Complexity
From fastest to slowest:
1. `StableSoftplusLipschitzLinear`/`GeometricMeanLipschitzLinear` O((`in_dim` $\times$ `out_dim`))
2. `SpectralNormalizedLinear` (O(`in_dim` $\times$ `out_dim`) + power iteration cost)
3. OrthogonalLipschitzLinear (O(`n_reflectors` $\times$ `in_dim` $\times$ `out_dim`))
4. JacobianNormLipschitzLinear (O(`num_samples` $\times$ `batch_size` $\times$ `in_dim` $\times$ `out_dim`))
