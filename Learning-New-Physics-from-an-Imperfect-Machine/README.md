# Learning New Physics from an Imperfect Machine
## Learn how to propagate systematic uncertainties in the Neyman-Pearson test statistic using machine learning

The exercise studies the 1D toy model proposed in *Learning new physics from an imperfect machine* [(Eur. Phys. J. C 82, 275 (2022)](https://doi.org/10.1140/epjc/s10052-022-10226-y).

The reference (e.g. anomaly-free) data distribution is an exponential distribution with uncertainty affecting the scale and normalization. Uncertainties are modeled by means of two nuisance parameters ($\nu_N,\,\nu_S$):
$$n(x|{\rm R}_{\nu_N,\nu_S})=n({\rm R_0})\exp[-xe^{-\nu_S}-\nu_S+\nu_N]$$


This folder contains two notebooks:
- `Parametric_1Dtoy.ipynb` (STEP 1): shows how to learn the parametriation of shape effects on the distribution. This is the first step to run. The parametric model learnt in this notebook is used as input in the second step of the procedure.
- `GOF_1Dtoy.ipynb` (STEP 2): implements the Neyman-Pearson goodness of fit (GOF) test with systematic uncertainties integrated by means of nuisance parameters.
