colors = {"Truth": "black",
          "OneLayer/":"gray",
          # Deterministic models
         "LinearRegression_N100/":"darkred",
         "NN_2layer_N100/":"midnightblue",
         # Dropout (epistemic - lighter, purple)
         "DropoutNN_2layer_N100/":"pink",
         "DropoutNN_2layer_N100/deterministic_":"magenta",
         # Maxlikelihood models (aleatoric - lighter, yellow)
         "AleatoricLinearRegression_N100/":"red",
         "AleatoricNN_2layer_N100/":"darkgoldenrod",
         "AleatoricNN_2layer_N100/deterministic_":"chocolate",
         # Bayesian models (epistemic - purple, aleatoric - green)
         "LinearRegression_N100/epistemic_":"orangered",
         "LinearRegression_N100/aleatoric_":"darkorange",
         "BayesianNN_2layer_N100/epistemic_":"darkorchid",
         "BayesianNN_2layer_N100/aleatoric_":"seagreen",
         "BayesianNN_2layer_N100/both_":"blue",
         "BayesianNN_2layer_N100/deterministic_":"midnightblue",
         "BayesianNN_2layer_N50/epistemic_":"darkorchid",
         "BayesianNN_2layer_N50/aleatoric_":"seagreen",
         "BayesianNN_2layer_N50/both_":"blue",
         "BayesianNN_2layer_N50/deterministic_":"midnightblue"
          }

labels = {"Truth":"Truth",
         "OneLayer/":"OneLayer",
         # Dropout
         "DropoutNN_2layer_N100/":"NN drp",
         "DropoutNN_2layer_N100/deterministic_":"NN drp det",
         # Deterministic models
         "LinearRegression_N100/":"LR",
         "NN_2layer_N100/":"NN",
         # Maxlikelihood models
         "AleatoricLinearRegression_N100/":"LR mle",
         "AleatoricNN_2layer_N100/":"NN mle",  
         "AleatoricNN_2layer_N100/deterministic_": "NN mle det",
         # Bayesian models
         "LinearRegression_N100/epistemic_":"LR epi",
         "LinearRegression_N100/aleatoric_":"LR ale",
         "BayesianNN_2layer_N100/epistemic_":"BNN epi",
         "BayesianNN_2layer_N100/aleatoric_":"BNN ale",
         "BayesianNN_2layer_N100/both_":"BNN both",
         "BayesianNN_2layer_N100/deterministic_":"BNN det",
         "BayesianNN_2layer_N50/epistemic_":"BNN epi",
         "BayesianNN_2layer_N50/aleatoric_":"BNN ale",
         "BayesianNN_2layer_N50/both_":"BNN both",
         "BayesianNN_2layer_N50/deterministic_":"BNN det"
}