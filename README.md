## Epistemic and Aleatoric Uncertainty in Weather and Climate Modelling

This is the repo for the code to explore epistemic and aleatoric uncertainty across timescales. 
We use the two layer Lorenz 1996 Model and learn a parameterisation for the small-scale variables.
We use a Bayesian neural network to separate uncertainties in epistemic (from the model parameters - weights, biases)
and aleatoric (from the subgrid variability in the training). Then we run this in an online setting to
compare how epistemic and aleatoric uncertainty evolve over weather and climate timescales.


