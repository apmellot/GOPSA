# GOPSA
# Geodesic Optimization for Predictive Shift Adaptation on EEG data

This repository contains the code and tools to reproduce the results obtained in the paper: [Geodesic Optimization for Predictive Shift Adaptation on EEG data](https://arxiv.org/pdf/2407.03878)

## Abstract
Electroencephalography (EEG) data is often collected from diverse contexts involving different populations and EEG devices. This variability can induce distribution shifts in the data $X$ and in the biomedical variables of interest $y$, thus limiting the application of supervised machine learning (ML) algorithms.
While domain adaptation (DA) methods have been developed to mitigate the impact of these shifts, such methods struggle when distribution shifts occur simultaneously in $X$ and $y$.
As state-of-the-art ML models for EEG represent the data by spatial covariance matrices,
which lie on the Riemannian manifold of Symmetric Positive Definite (SPD) matrices, it is appealing to study DA techniques operating on the SPD manifold.
This paper proposes a novel method termed Geodesic Optimization for Predictive Shift Adaptation (GOPSA) to address test-time multi-source DA for situations in which source domains have distinct $y$ distributions.
GOPSA exploits the geodesic structure of the Riemannian manifold to jointly learn a domain-specific re-centering operator representing site-specific intercepts and the regression model.
We performed empirical benchmarks on the cross-site generalization of age-prediction models with resting-state EEG data from a large multi-national dataset (HarMNqEEG), which included $14$ recording sites and more than $1500$ human participants.
Compared to state-of-the-art methods, our results showed that GOPSA achieved significantly higher performance on three regression metrics ($R^2$, MAE, and Spearman's $\rho$) for several source-target site combinations, highlighting its effectiveness in tackling multi-source DA with predictive shifts in EEG data analysis.
Our method has the potential to combine the advantages of mixed-effects modeling with machine learning for biomedical applications of EEG, such as multicenter clinical trials.

## Citation
```
@article{mellot2024geodesic,
  title={Geodesic Optimization for Predictive Shift Adaptation on EEG data},
  author={Mellot, Apolline and Collas, Antoine and Chevallier, Sylvain and Gramfort, Alexandre and Engemann, Denis A},
  journal={arXiv preprint arXiv:2407.03878},
  year={2024}
}
```
