## Epistemic and Aleatoric Uncertainty in Weather and Climate Modelling

This is the repo for the code to explore epistemic and aleatoric uncertainty across timescales. 
We use the two layer Lorenz 1996 Model and learn a parameterisation for the small-scale variables.
We use a Bayesian neural network to separate uncertainties in epistemic (from the model parameters - weights, biases)
and aleatoric (from the subgrid variability in the training). Then we run this in an online setting to
compare how epistemic and aleatoric uncertainty evolve over weather and climate timescales.


### Summary 

![Uncertainties on Weather/Climate Timescales](assets/GRAPHICAL_ABSTRACT.tif)

Aleatoric and epistemic uncertainties over time on weather and climate timescales, estimated through ensembles that sample aleatoric and epistemic uncertainty using Bayesian neural networks for parameterisations in the Lorenz 1996 model. The spread shows the 16th and 84th percentiles. For weather, we show divergence of large-scale variables from the ensemble mean, where aleatoric uncertainty from subgrid v   ariability dominates. For climate, we show the ratio of time spent in one weather regime, where epistemic uncertainty is the dominant source of uncertainty.

### Project Directory Overview

L96/
Core implementation of the Lorenz-96 dynamical system, including the model equations and numerical integration methods.

tests/
Unit tests for verifying correctness of the core Lorenz-96 model and related utilities.

ml_models/
PyTorch and Pyro models for neural network and Bayesian neural network parameterisations.

scripts/
Reusable functions used for generating training data, training ML models and running L96 once coupled to parameterisation with different sampling methods.

experiments/
Runs scripts to generate data, train ML models, and run coupled models. 

plotting_scripts/
Reusable functions to create custom plots for analysis, including error trajectories, spread-skill plots, probability distribution functions, and animations.

create_plots/
Collection of scripts that generate all plots and visualizations used in analysis, the paper and presentations.

utils/
General purpose utility functions for metrics, probability scoring, plotting helpers, file concatenation, and data transformation.

## Authors

Laura Mansfield (primary developer) and Hannah Christensen
Email: laura.mansfield@physics.ac.uk


If you use this repository or build upon it, feel free to reach out or open an issue. Contributions and feedback are welcome.
