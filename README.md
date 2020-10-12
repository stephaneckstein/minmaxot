
#  MinMax Methods for Optimal Transport and Beyond: Regularization, Approximation and Numerics

This repository explains how to reproduce the numerical experiments in Section 5 of the paper "MinMax Methods for Optimal Transport and Beyond: Regularization, Approximation and Numerics".

## Requirements

To run the code, we used python 3.7 and anaconda. The only non-default packages necessary are tensorflow 1.15.0, keras and matplotlib.
An anaconda environment.yml file and requirements.txt file are included.
For the setup of the environment, if using anaconda, run


```setup
conda env create -f environment.yml
```

or, using pip, run

```setup
pip install -r requirements.txt
```

## Running the experiments

To solve the optimization problems in Section 5.1., run the respective programs base_case.py, base_case10.py, divergence_reg.py, divergence_reg10.py, lipschitz_reg.py, lipschitz_reg10.py within the folder dcot. E.g., within the project folder, run

```optimization1
python dcot/base_case.py
```
The results will be saved in dcot/output. We provided the results from our runs which lead to the values reported in the paper.

For the optimization problems in Section 5.2., run the programs base_case.py, mixture_case.py, unrolling_case.py, combined_case.py within the folder mot. E.g., within the project folder, run

```optimization2
python mot/base_case.py
```
The results will be saved in mot/output. We provide partially the results (1 out of 10 runs due to space constraints) which lead to the values reported in the paper.


## Evaluation

To create Figure 1 illustrating the results from Section 5.1., run

```eval1
python dcot/figure1.py
```

To create Table 1 illustrating the results from Section 5.2., run

```eval2
python mot/table1.py
```
