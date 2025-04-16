colors = {"Truth": "black",
          "OneLayer":"gray",
          # Deterministic models
         "LinearRegression_N100":"darkred",
         "NN_2layer_N100":"darkblue",
         # Dropout
         "DropoutNN_2layer_N100/":"blueviolet",
         # Maxlikelihood models
         "AleatoricLinearRegression_N100":"red",
         "AleatoricNN_2layer_N100/":"blue",  
         # Bayesian models
         "LinearRegression_N100/epistemic_":"orangered",
         "LinearRegression_N100/aleatoric_":"darkorange",
         "BayesianNN_2layer_N100/epistemic_":"royalblue",
         "BayesianNN_2layer_N100/aleatoric_":"cornflowerblue",
          }

labels = {"Truth":"Truth",
         "OneLayer":"OneLayer",
         # Dropout
         "DropoutNN_2layer_N100/":"NN drp",
         # Deterministic models
         "LinearRegression_N100":"LR",
         "NN_2layer_N100":"NN",
         # Maxlikelihood models
         "AleatoricLinearRegression_N100":"LR mle",
         "AleatoricNN_2layer_N100/":"NN mle",  
         # Bayesian models
         "LinearRegression_N100/epistemic_":"LR epi",
         "LinearRegression_N100/aleatoric_":"LR ale",
         "BayesianNN_2layer_N100/epistemic_":"BNN epi",
         "BayesianNN_2layer_N100/aleatoric_":"BNN ale",
}