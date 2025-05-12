# Lipschitz-Regularized SegFormer (LRSegformer)

## Overview

This repository contains a variant of the SegFormer architecture - a Transformer-based neural network widely recognized for its high efficiency and accuracy in semantic segmentation and image classification tasks. Here, we introduce the Lipschitz-Regularized SegFormer (LRSegformer), enhancing the original architecture by incorporating novel Lipschitz-constrained linear layers into the MLP decoder head to improve model robustness, generalization, and stability.

This experiment was extended from the use of the Segformer variant regularized with the geometric mean of Lipschitz constants in my thesis experiments. The thesis document for "Beyond Clear Paths" is hosted as [open access at Proquest](https://www.proquest.com/docview/3155972317).

> [!NOTE] Results will be posted after full training of each variant in the future.

---

## SegFormer Overview

SegFormer is a hierarchical Transformer-based architecture with a lightweight multilayer perceptron (MLP) decoder head, designed specifically for semantic segmentation. Key innovations include hierarchical multi-scale feature extraction, avoidance of positional encoding for resolution invariance, and a compact yet powerful MLP decoder that aggregates local and global features effectively.

**Paper:** ["SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"](https://arxiv.org/abs/2105.15203)

---

## LRSegformer: Lipschitz Constraints for Robust Generalization

The Lipschitz-Regularized SegFormer employs Lipschitz constraints primarily in the MLP decoder layers, thereby regularizing the network and controlling how quickly the network's output can change. This provides smoother and more stable feature representations, enhancing robustness against adversarial examples and improving generalization.

The Segformer implementations within this repository (for now) simply inherit from the implementations within HuggingFace's [`transformers.models.segformer`](https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/segformer) module. This new implementation includes both the extended semantic segmentation models with the new regularized linear layers of the MLP decoder head and the simpler image classification models, which uses a single linear layer for its output classifier.

### Motivation:

* **Robustness to Adversarial Perturbations:** Lipschitz constraints directly reduce sensitivity to input variations.
* **Generalization and Regularization:** Controlled Lipschitz constants imply smoother function spaces, mitigating overfitting and leading to better generalization.
* **Stable Training Dynamics:** Lipschitz continuity prevents exploding or vanishing gradients, particularly in deep Transformer architectures.

Implementing Lipschitz constraints improves network robustness, reduces sensitivity to adversarial attacks, and stabilizes training. Practically, tighter Lipschitz constraints (spectral normalization, orthogonal initialization) yield stronger theoretical guarantees (reduced VC-dimension, better generalization), while softer constraints (StableSoftplus, GeometricMean) balance expressivity with improved regularization.

---

## Implemented Lipschitz-Constrained Layers

The LRSegformer implements multiple Lipschitz-constrained linear layers, summarized briefly here and detailed extensively in `Lipschitz Layer Formulations.md`:

| Method                            | Lipschitz Constant            | Pros                                         | Cons                                           |
| --------------------------------- | ----------------------------- | -------------------------------------------- | ---------------------------------------------- |
| **StableSoftplusLipschitzLinear** | Learned softplus-based scalar | Differentiable, numerically stable           | Loose bound (L1 norm), less expressive         |
| **GeometricMeanLipschitzLinear**  | Learned geometric mean scalar | Numerically stable, geometric mean scaling   | Slightly looser bound, potential rank collapse |
| **SpectralNormalizedLinear**      | Spectral norm = 1             | Tightest Lipschitz bound, good stability     | Computational overhead (power iterations)      |
| **OrthogonalLipschitzLinear**     | Exact orthonormal (=1)        | Exact norm preservation, excellent stability | Limited expressivity, dimensional constraints  |

**Note:** The previously attempted Jacobian-based approach (`JacobianNormLipschitzLinear`) has been temporarily removed due to computational inefficiency and limited advantage in linear layers.

---

## Experimental Goal and Setup

The primary goal of this experiment is to demonstrate how various Lipschitz constraints impact the generalization capability and robustness of SegFormer in semantic segmentation and classification tasks. Experiments are conducted on standard benchmarks:

* **Semantic Segmentation:** ADE20K (`zhoubolei/scene_parse_150`)
* **Classification:** ImageNet-1K (`ILSVRC/imagenet-1k`)

Models are pre-trained using the smallest available standard SegFormer weights (`nvidia/mit-b0`), which were pre-trained on ImageNet-1k, and fine-tuned under different Lipschitz constraints.

Generalization benchmark datasets are still under consideration.

Performance metrics to be tracked include mIoU, Accuracy + Precision + Recall, F1 scores, and the Matthews Correlation Coefficient.

---

## How to Run Experiments

To reproduce or extend the experiments, ensure dependencies (`torch`, `torchvision`, `transformers`, `datasets`, and `torchmetrics`) are installed, and use the provided CLI:

```bash
python run.py --task segmentation --variant lipschitz --strategy spectral_norm --epochs 30 --batch_size 8 --lr 5e-5
```
> [!NOTE] This project may never be implemented as a true Python library, i.e. runnable with `python -m lrsegformer`, as it's primarily an experiment and (some of) the novel layers will hopefully be extended and merged to core Pytorch in the future.

Example parameters:
* `--task`: segmentation/classification
* `--variant`: baseline/lipschitz
* `--strategy`: geometric\_mean, spectral\_norm, stable\_softplus, orthogonal
* `--lambda_lip`: Regularization strength for Lipschitz penalty (default: `0.1`)

---

## Lipschitz Layer Summaries

1. **StableSoftplusLipschitzLinear:**  Uses a learnable scalar (`c`) parameterizing the Lipschitz constant via a numerically stable softplus. It is differentiated smoothly but loosely bounds Lipschitz continuity via an L1 norm. Note that this may be replaced with the infinity norm following [Liu et al. 2022](https://arxiv.org/abs/2202.08345).

2. **GeometricMeanLipschitzLinear:**  Employs geometric mean scaling for numerically stable parameterization of the Lipschitz constant. It provides stable gradients but can lead to overly restrictive constraints if poorly initialized.

3. **SpectralNormalizedLinear:**  Directly constrains the Lipschitz constant using the exact spectral norm of the weight matrix. It offers the tightest bound and excellent stability at the cost of computational overhead due to power iterations during training.

4. **OrthogonalLipschitzLinear:**  Parameterizes weights using Householder reflections ensuring exact orthogonality and hence an exact Lipschitz constant of 1.0. It guarantees norm preservation and stability but limits expressivity due to the orthonormal constraint.

More detailed formulations (under construction) can be found in the `docs/` folder.

---

## Repository Structure

```plaintext
.
├── models/
│   ├── lipschitz_linear.py       # Lipschitz layer implementations
│   ├── segformer.py              # Lipschitz-integrated SegFormer architecture
│   └── model_registry.py         # Factory methods for model instantiation
├── experiments/
│   ├── run.py                    # Main experiment runner
│   ├── train.py                  # Training loop and utilities
│   ├── loading.py                # Dataset loading and preparation
│   └── metrics.py                # Metrics tracking and evaluation utilities
├── experiments/
│   ├── layer_formulations.md     # Detailed math/formulation docs
│   └── lipschitz_constraints.md  # Theoretical discussions on constraints
└──
```

---

## Future Work and Contributions

* Integrate Lipschitz constraints into Transformer encoder layers (e.g., attention mechanisms).
* Explore implicit and Jacobian-based Lipschitz constraints applied to nonlinear activations with improved efficiency.
* Conduct extensive benchmarking against standard adversarial attack frameworks to quantify robustness improvements.

---

## License

This repository extends the SegFormer architecture under the Apache License 2.0. Original SegFormer license applies, with extended contributions following the same license.

