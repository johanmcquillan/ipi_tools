# i-PI Tools

This is a small Python package written to apply an external forcefield to an i-PI simulation [1]. Specifically, they are designed to apply a one-dimensional potential to one or more molecular layers of water. This is to simulate the affect of two hydrophobic surfaces (such as graphene) confining the water such that it behaves quasi-two-dimensionally [2].

The purpose of this was to determine if the machine learning water potential MB-pol [3] (which has been trained on cluster and three-dimensional bulk ab initio data) is accurate in the two-dimensional case.

### Prerequisites

This package is written in Python 2.7. ```numpy``` and ```scipy``` are required. ```matplotlib``` is needed to visualise the confinement potentials.

## References

1. M. Ceriotti, J. More, D. E. Manolopoulos, Comp. Phys. Comm., **185**, 3, 1019-1026 (2014)
2. W. Zhao, L. Wang, J. Bai, L. Yuan, J., Yang, X. C. Zeng, Acc. Chem. Res., **47**, 8, 2505-2513 (2014)
3. G. R. Medders, A. W. Götz, M. A. Morales, P. Bajaj, F. Paesani, J. Chem. Phys., **143**, 10, 104102 (2015)

## Acknowledgments

* Prof. Angelos Michaelides, University College London, London Centre for Nanotechnology, for supervising this research (https://www.ucl.ac.uk/catalytic-enviro-group/).

* Dr. Wei Fang, currently of ETH Zürich, for his expertise with i-PI (http://www.richardson.ethz.ch/people/person-detail.html?persid=245651).
