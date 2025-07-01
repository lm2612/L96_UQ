plot_dict = { 
    "Truth" : {
        "color":"black",
        "label":"Truth",
        "path":"Truth/",
        },
    "OneLayer" : {
        "color":"gray",
        "label":"OneLayer",
        "path":"OneLayer/",
        },
    "LinearRegression": {
        "color":"gray",
        "label":"LR",
        "path":"LinearRegression_N{N_train}/",
        },
    "NN" : {
        "color":"midnightblue",
        "label":"NN",
        "path":"NN_2layer_N{N_train}/",
        },  
    "DropoutNN" : {
        "color":"pink",
        "label":"NN drp",
        "path":"DropoutNN_2layer_N{N_train}/",
        },
    "EnsNN" : {
        "color":"deeppink",
        "label":"NN ens",
        "path":"NN_2layer_N{N_train}_seed{seed}",
        },
    "AleatoricNN" : {
        "color":"darkgoldenrod",
        "label":"NN mle",
        "path":"AleatoricNN_2layer_N{N_train}/",
        },
    "BayesianNN" : {
        "color":"blue",
        "label":"BNN both",
        "path":"BayesianNN_2layer_N{N_train}/both_",
        },
    "BayesianNNepistemic" : {
        "color":"darkorchid",
        "label":"BNN epi",
        "path":"BayesianNN_2layer_N{N_train}/epistemic_",
        },
    "BayesianNNaleatoric" : {
        "color":"seagreen",
        "label":"BNN ale",
        "path":"BayesianNN_2layer_N{N_train}/aleatoric_",
        },
}

              


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
         "BayesianNN_2layer_N10/epistemic_":"darkorchid",
         "BayesianNN_2layer_N10/aleatoric_":"seagreen",
         "BayesianNN_2layer_N10/both_":"blue",
         "BayesianNN_2layer_N10/deterministic_":"midnightblue",
         "BayesianNN_multivariate_2layer_N50/epistemic_":"darkorchid",
         "BayesianNN_2layer_N50/epistemic_":"darkorchid",
         "BayesianNN_multivariate_2layer_N50/aleatoric_":"seagreen",
         "BayesianNN_2layer_N50/aleatoric_":"seagreen",
         "BayesianNN_multivariate_2layer_N50/both_":"blue",
         "BayesianNN_2layer_N50/both_":"blue",
         "BayesianNN_2layer_N50/deterministic_":"midnightblue",
         "BayesianNN_2layer_N500/epistemic_":"darkorchid",
         "BayesianNN_2layer_N500/aleatoric_":"seagreen",
         "BayesianNN_2layer_N500/both_":"blue",
         "BayesianNN_2layer_N1000/epistemic_":"darkorchid",
         "BayesianNN_2layer_N1000/aleatoric_":"seagreen",
         "BayesianNN_2layer_N1000/both_":"blue",
         "NN_2layer_online_Ntime2_seed100/":"midnightblue",
         "NN_2layer_online_Ntime10_seed100/":"blue",
         "NN_2layer_online_Ntime50_seed100/":"lightblue",
         "NN_2layer_online_Ntime100_seed100/":"skyblue",
         "NN_2layer_online_Ntime200_seed100/":"turquoise",
         "NN_2layer_regime0_N50/longrun_":"blue",
         "NN_2layer_regime1_N50/longrun_":"red",

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
         "BayesianNN_2layer_N10/epistemic_":"BNN epi",
         "BayesianNN_2layer_N10/aleatoric_":"BNN ale",
         "BayesianNN_2layer_N10/both_":"BNN both",
         "BayesianNN_2layer_N10/deterministic_":"BNN det",
         "BayesianNN_2layer_N50/epistemic_":"BNN epi",

         "BayesianNN_2layer_N50/aleatoric_":"BNN ale ",
         "BayesianNN_2layer_N50/both_":"BNN both",
         "BayesianNN_2layer_N50/epistemic_":"BNN epi",

         "BayesianNN_multivariate_2layer_N50/aleatoric_":"BNNmv ale ",
         "BayesianNN_multivariate_2layer_N50/both_":"BNNmv both",
         "BayesianNN_multivariate_2layer_N50/epistemic_":"BNNmv epi",
         "BayesianNN_2layer_N500/aleatoric_":"BNN ale",
         "BayesianNN_2layer_N500/both_":"BNN both",
         "BayesianNN_2layer_N500/epistemic_":"BNN epi",
         "BayesianNN_2layer_N1000/aleatoric_":"BNN ale",
         "BayesianNN_2layer_N1000/both_":"BNN both",
         "BayesianNN_2layer_N1000/epistemic_":"BNN epi",
         "NN_2layer_online_Ntime2_seed100/":"online N=1",
         "NN_2layer_online_Ntime10_seed100/":"online N=10",
         "NN_2layer_online_Ntime50_seed100/":"online N=50",
         "NN_2layer_online_Ntime100_seed100/":"online N=100",
         "NN_2layer_online_Ntime200_seed100/":"online N=200",
         "NN_2layer_regime0_N50/longrun_":"Regime 1",
         "NN_2layer_regime1_N50/longrun_":"Regime 2",

}


def plotcolor(name):
    if "aleatoric_AR1" in name:
        return "lightgreen"
    elif "aleatoric" in name:
        return "seagreen"
    elif "epistemic" in name:
        return "darkorchid"
    elif "both" in name:
        return "blue"
    elif "deterministic" in name:
        return "midnightblue"
    elif "Truth" in name:
        return "black"
    elif "Dropout" in name:
        return "pink"
    elif "Aleatoric" in name:
        return "darkgoldenrod"
    