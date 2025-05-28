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


## 3. Cite our work
```text
@inproceedings{
    jalali2025spec,
    title={Towards an Explainable Comparison and Alignment of Feature Embeddings},
    author={Mohammad Jalali and Bahar Dibaei Nia and Farzan Farnia},
    booktitle={Forty-second International Conference on MachineLearning},
    year={2025},
    url={https://openreview.net/forum?id=Doi0G4UNgt}
}
```


