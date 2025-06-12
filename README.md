# embedding-comparison
[ICML 2025] Official implementation of SPEC method for interpretable embedding comparison.

paper: Towards an Explainable Comparison and Alignment of Feature Embeddings


[Mohammad Jalai](https://mjalali.github.io/) <sup>1</sup>, Bahar Dibaei Nia <sup>2</sup>,
[Farzan Farnia](https://www.cse.cuhk.edu.hk/people/faculty/farzan-farnia/) <sup>1</sup>

<sup>1</sup> <sub>**The Chinese University of Hong Kong (CUHK)**</sub>

<sup>2</sup> <sub>**Sharif University of Technology**</sub>

## Abstract

Feature embedding models have been widely applied to many downstream tasks across various domains. While several standard feature embeddings have been developed in the literature, comparisons of these embeddings have largely focused on their numerical performance in classification-related downstream applications. However, an interpretable comparison of different embeddings requires identifying and analyzing mismatches between similar sample groups clustered within the embedding spaces. In this work, we propose the \emph{Spectral Pairwise Embedding Comparison (SPEC)} framework to compare embeddings and identify their differences in clustering a benchmark dataset. Our approach examines the kernel matrices derived from two embeddings and leverages the eigendecomposition of the differential kernel matrix to detect sample clusters that are captured differently by the two embeddings. We present a scalable implementation of this kernel-based approach, with computational complexity that grows linearly with the sample size. Furthermore, we introduce an optimization problem using this framework to align two embeddings, enhancing interpretability by correcting potential biases and mismatches, and ensuring that clusters identified in one embedding are also captured in the other. Finally, we provide numerical results showcasing the application of the SPEC framework to compare and align standard embeddings on large-scale datasets such as ImageNet and MS-COCO.

## Overview of SPEC

![SPEC Method](./SPEC_method.png)

## What is SPEC?

Modern embedding models—such as CLIP and DINOv2 deliver strong downstream accuracy, yet numerical scores alone do not explain *how* the embeddings differ. SPEC fills this gap.

Given two embeddings, SPEC

1. Computes a kernel matrix for each embedding over a reference dataset.  
2. Forms their *difference* kernel.  
3. Spectrally decomposes that difference.  

Each eigenvector identifies a cluster of samples that one embedding groups but the other does not; the associated eigenvalue quantifies the strength of that mismatch.

The naïve eigen-solve scales as O(n³). SPEC avoids that cost: for bounded feature dimension *d* (or its random-Fourier proxy) the key eigenspace is recoverable in O(max{d³, n}) time—linear in sample size for practical *d* values.

---

## Project Structure

```

embedding-comparison/
├── spec-core/       # SPEC routines: kernels, eigensolvers, visualisation
└── spec-align/      # SPEC-align: gradient-based embedding alignment

````
---

## Installation

```bash
git clone https://github.com/yourusername/embedding-comparison.git
cd embedding-comparison
pip install -r requirements.txt
````

---

## Quick Start: comparing two embeddings

### 1. Load features

```python
import numpy as np
import torch
from spec_core.gaussian import gaussian_covariance
from spec_core.diffembed import DiffEmbed

# Each .npz holds an array of shape [N, D]
clip = np.load('features/clip_imagenet.npz')['features']
dino = np.load('features/dino_imagenet.npz')['features']
```

---

### 2. Choose kernel bandwidths (σ values)

The Gaussian-kernel bandwidth σ controls how fast similarity decays with distance.

* If σ is **too small**, the kernel becomes nearly diagonal; every point looks unique and clustering signals disappear.
* If σ is **too large**, the kernel approaches a constant matrix; meaningful structure is washed out.

**Practical procedure**

1. For each embedding, pick an initial σ (for cosine-distance features a rule-of-thumb is the median pairwise distance).
2. Compute the approximate covariance via random Fourier features (step 3 below).
3. Plot the eigenvalues of each covariance.
4. Adjust σ per embedding until

   * their top eigenvalues are within the same order of magnitude, and
   * the spectra decay smoothly (indicating separable clusters).

Example working values:

```python
sigma_clip = 3.5
sigma_dino = 25.0
```

---

### 3. Compute Gaussian covariances

```python
cov_clip, _, phi_clip = gaussian_covariance(
    torch.from_numpy(clip).float(),
    rff_dim=2000,
    batchsize=128,
    sigma=sigma_clip,
    return_features=True)

cov_dino, _, phi_dino = gaussian_covariance(
    torch.from_numpy(dino).float(),
    rff_dim=2000,
    batchsize=128,
    sigma=sigma_dino,
    return_features=True)
```

`phi_clip` and `phi_dino` are the random-Fourier features that proxy the kernels.

---

### 4. Run SPEC

```python
spec = DiffEmbed(sigma=0)
eigenvalues, eigenvectors = spec.DiffEmbed_by_covariance_matrix(
    x=clip,
    y=dino,
    cov_function=None,
    phi_x=phi_clip,
    phi_y=phi_dino,
    eta=1)
```

`eigenvalues` quantify cluster mismatches; `eigenvectors` weight the samples in each mismatched cluster.

---

### 5. Visualise cluster differences

```python
from spec_core.visualize import visualize_modes_covariance
from spec_core.dataset import ImageFilesDataset

image_paths = np.load('paths/imagenet_paths.npy')
dataset = ImageFilesDataset(path='', name='imagenet-val',
                            path_files=image_paths)

visualize_modes_covariance(
    eigenvalues=eigenvalues,
    eigenvectors=eigenvectors,
    x_feature=phi_clip,
    y_feature=phi_dino,
    num_visual_mode=10,
    num_samples_per_mode=20,
    save_dir='outputs/clip_vs_dino/',
    dataset=dataset,
    save_file=True,
    x=clip,
    y=dino,
    model_names=('CLIP', 'DINOv2')
)
```

Each output montage highlights the images most responsible for a particular embedding mismatch.

---

## Contributions

* SPEC: kernel-difference eigendecomposition for explainable embedding comparison
* Linear-time computation via random Fourier features
* SPEC-diff: spectral distance measuring maximal cluster mismatch
* SPEC-align: gradient-based method to align embeddings (in `spec-align/`)

---

## Cite our work

```bibtex
@inproceedings{
    jalali2025spec,
    title={Towards an Explainable Comparison and Alignment of Feature Embeddings},
    author={Mohammad Jalali and Bahar Dibaei Nia and Farzan Farnia},
    booktitle={Forty-second International Conference on MachineLearning},
    year={2025},
    url={https://openreview.net/forum?id=Doi0G4UNgt}
}

```




