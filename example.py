import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from termcolor import colored
from datetime import datetime

# Scikit-learn
from sklearn import preprocessing
from sklearn.utils import check_random_state

# torch
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from mvpb import MultiViewMajorityVoteLearner, MajorityVoteLearner
from mvpb.util import uniform_distribution


# Import data
from data import (SampleData,
                           Nhanes,
                           MultipleFeatures,
                           MNIST_MV_Datasets,
                           Fash_MNIST_MV_Datasets,
                           EMNIST_Letters_MV_Datasets,
                           Mushrooms,
                           PTB_XL_plus,
                           Nutrimouse,
                           ReutersEN,
                           IS,
                           CorelImageFeatures,
                           NUS_WIDE_OBJECT,
                           ALOI,
                           train_test_split,
                           train_test_merge,
                           s1_s2_split,
                           multiclass_to_binary,
                           balance_dataset,
                           other_binary_options,
                           poison_dataset)

##############################################################
dataset = MultipleFeatures(size="large")
X_train, y_train, X_test, y_test = dataset.get_data()
if isinstance(dataset, PTB_XL_plus):
    real_classes = dataset.get_real_classes(np.unique(y_train))

Xs_train = []
Xs_test = []
for xtr, xts in zip(X_train, X_test):
    scaler = preprocessing.MinMaxScaler().fit(xtr)
    Xs_train.append(scaler.transform(xtr))
    Xs_test.append(scaler.transform(xts))

X_train_concat = [np.concatenate(Xs_train, axis=1)]
X_test_concat = [np.concatenate(Xs_test, axis=1)]


##############################################################
RUNS = range(1)

OPTIMIZE_LAMBDA_GAMMA = True
# ALPHA = [1, 0.5, 1.1, 2]
ALPHA = [1.1]
MAX_ITER = 1000

stump_config = {
    "name": "stump",
    "n_estimators": 100,
    "max_depth": 1,
    "max_features": 0.5,
}
weak_learners_config = {
    "name": "weak_learner",
    "n_estimators": 100,
    "max_depth": 3,
    "max_features": 0.5,
}
strong_learners_config = {
    "name": "strong_learner",
    "n_estimators": 100,
    "max_depth": 6,
    "max_features": 0.8,
}

# CFG = [stump_config, weak_learners_config, strong_learners_config]
CFG = [weak_learners_config]

EPOCHS = 15

TO_BINARY  = "ovo" # One of ["ovr", "ovo", "other",  None]
label_1 = 4
# if isinstance(dataset, PTB_XL_plus):
#     label_1 = np.unique(y_test)[np.where(real_classes == "['NORM']")[0][0]]
label_2 = 9

POISON = False

USE_UNLABELED = False
s_labeled_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] if USE_UNLABELED else [0.1]

# BOUNDS = ['PBkl', 'PBkl_inv', 'TND_DIS', 'TND_DIS_inv', 'TND', 'TND_inv', 'DIS', 'DIS_inv', 'Cbound', 'C_TND']
BOUNDS = ['Cbound']
MV_SPECIFIC_BOUNDS = ['TND_DIS', 'TND_DIS_inv']

m = y_train.size #350
test_size = 1 - (m  / (y_test.size+y_train.size))
experiments = {}
for s_labeled_size in s_labeled_sizes:
    experiments[s_labeled_size] = {}
    for alpha in ALPHA:
        experiments[s_labeled_size][alpha] = {}
        for cfg in CFG:
            experiments[s_labeled_size][alpha][cfg["name"]] = []


##############################################################
# Transform to binary OVR (One Vs Rest) or OVO (One Vs One) if needed
if TO_BINARY == "ovr":
    Xs_train, y_train, Xs_test, y_test = multiclass_to_binary(Xs_train, y_train, Xs_test, y_test, type=TO_BINARY, label_1=label_1)
elif TO_BINARY == "ovo":
    Xs_train, y_train, Xs_test, y_test = multiclass_to_binary(Xs_train, y_train, Xs_test, y_test, type=TO_BINARY, label_1=label_1, label_2=label_2)
elif TO_BINARY == "other":
    y_train, y_test = other_binary_options(dataset, y_train, y_test)
else:
    print(colored(f"WARNING: TO_BINARY={TO_BINARY}, continuing", 'yellow'))


##############################################################
before_merge = (Xs_train, y_train, Xs_test, y_test)
Xs, y = train_test_merge(Xs_train, y_train, Xs_test, y_test)
os.makedirs("results", exist_ok=True)
    

# iterate over the labeled data sizes #
for i, s1_size in enumerate(s_labeled_sizes):
    print(colored(f"############ Using {s1_size*100}% labeled data ############", 'black', on_color='on_blue'))
    s_labeled_dir = 'results'+f"/s_labeled-{int(s1_size*100)}"
    os.makedirs(s_labeled_dir, exist_ok=True)
### iterate over the alpha values ###
    
    for j, alpha in enumerate(ALPHA):
        print(colored(f"\t############ Using {alpha=} ############", 'black', on_color='on_blue'))
        alpha_dir = s_labeled_dir+ f"/alpha-{alpha}"
        os.makedirs(alpha_dir, exist_ok=True)
        
#### iterate over the configurations ####
        for k, config in enumerate(CFG):
            print(colored(f"\t\t############ Using {config['name']} ############", 'black', on_color='on_blue'))
            for run in RUNS:
                print(colored(f"\n----------------Run {run+1}---------------", 'blue'))

                # Shuffle and split the dataset into training and testing
                # if not dataset.split:
                Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y,
                                              test_size=test_size, random_state=run*(i+1)*(j+1)*(k+1))
                # else:
                # Xs_train, y_train, Xs_test, y_test = before_merge

                # Split the dataset into labeled and unlabeled
                Xs_train, y_train, UlX, _ = s1_s2_split(Xs_train, y_train,
                                                s1_size=s1_size, random_state=run*(i+1)*(j+1)*(k+1))
                X_train_concat = np.concatenate(Xs_train, axis=1)
                X_test_concat = np.concatenate(Xs_test, axis=1)
                    
                # instantiate multiview dNDF classifier
                dNDF_mv = MultiViewMajorityVoteLearner(nb_estimators=config["n_estimators"],
                                                        nb_views=len(Xs_train),
                                                        depth =config["max_depth"],
                                                        used_feature_rate=config["max_features"],
                                                        random_state=run,
                                                        epochs=EPOCHS,
                                                        use_dndf=False)
                
                # instantiate dNDF classifier for separate views and concatenated view
                dNDF_per_view = []
                for v in range(len(Xs_train)+1):
                    dNDF_per_view.append(MajorityVoteLearner(nb_estimators=config["n_estimators"],
                                                            depth =config["max_depth"],
                                                            used_feature_rate=config["max_features"],
                                                            random_state=run,
                                                            epochs=EPOCHS,
                                                            use_dndf=False))
                
                print("Training multiview classifier-------------------------------")
                dNDF_mv = dNDF_mv.fit(Xs_train, y_train)
                
                print("Training separate views classifiers-------------------------------")
                for v in range(len(Xs_train)):
                    dNDF_per_view[v] = dNDF_per_view[v].fit(Xs_train[v], y_train)

                print("Training concatenated view classifier-------------------------------")
                dNDF_per_view[-1] = dNDF_per_view[-1].fit(X_train_concat, y_train)
                
                
                # Optimize the posterior distributions for the each bound
                for bound in BOUNDS:
                    # Clear the posteriors (reset to uniform distribution)
                    dNDF_mv.clear_posteriors()
                    for v in range(len(Xs_train)):
                        dNDF_per_view[v].clear_posteriors()
                    
                    # use the unlabeled data for DIS
                    unlabeled_data, c_unlabeled_data = None, None
                    if USE_UNLABELED and bound in ['DIS', 'DIS_inv', 'TND_DIS', 'TND_DIS_inv',]:
                        unlabeled_data = UlX
                        c_unlabeled_data = np.concatenate(UlX, axis=1)
                        
                    if bound != "Uniform":
                        _, gibbs_risk, _ = dNDF_mv.mv_risk((Xs_train, y_train), incl_oob=False)
                        print(f"### Multiview classifier gibbs risk before Optim: {gibbs_risk}")
                        print(colored(f"Optimizing {bound} for multiview classifier-------------------------------", 'green'))
                        prev_time = datetime.now()
                        posterior_Qv , posterior_rho = dNDF_mv.optimize_rho(bound,
                                                            labeled_data=(Xs_train, y_train),
                                                            unlabeled_data=unlabeled_data,
                                                            incl_oob=False,
                                                            max_iter=MAX_ITER,
                                                            optimise_lambda_gamma=OPTIMIZE_LAMBDA_GAMMA,
                                                            alpha=alpha)
                        print(colored(f"Optimization took {datetime.now() - prev_time} -------------------------------", 'yellow'))
                        if bound not in MV_SPECIFIC_BOUNDS:
                            print(colored(f"Optimizing {bound} for separate views classifiers-------------------------------", 'green'))
                            posterior_Qs = []
                            for v in range(len(Xs_train)):
                                posterior_Q = dNDF_per_view[v].optimize_Q(bound,
                                                            labeled_data=(Xs_train[v], y_train),
                                                            unlabeled_data=unlabeled_data[v] if unlabeled_data else None,
                                                            incl_oob=False,
                                                            max_iter=MAX_ITER,
                                                            optimise_lambda_gamma=OPTIMIZE_LAMBDA_GAMMA,
                                                            alpha=1)
                                posterior_Qs.append(posterior_Q)
                            print(colored(f"Optimizing {bound} for concatenated classifier-------------------------------", 'green'))
                            posterior_Q_concat = dNDF_per_view[-1].optimize_Q(bound,
                                                            labeled_data=(X_train_concat, y_train),
                                                            unlabeled_data=c_unlabeled_data,
                                                            incl_oob=False,
                                                            max_iter=MAX_ITER,
                                                            optimise_lambda_gamma=OPTIMIZE_LAMBDA_GAMMA,
                                                            alpha=1)
                            posterior_Qs.append(posterior_Q_concat)
                        
                        _, gibbs_riska, _ = dNDF_mv.mv_risk((Xs_train, y_train), incl_oob=False)
                        print(f"### Multiview classifier gibbs risk after Optim: {gibbs_riska}")
                        # Compute the bound for the multiview classifier
                        print(colored(f"Optimization is done! -------------------------------", 'green'))
                    
                    print(colored(f"Computing the bound values ans risks -------------------------------", 'green'))
                    mv_bound, r1, r2, klqp, klrpi,  n1, n2 = dNDF_mv.bound(
                                        bound=bound,
                                        labeled_data=(Xs_train, y_train),
                                        unlabeled_data=unlabeled_data,
                                        incl_oob=False,
                                        alpha=alpha)
                    # Compute the risk of the multiview classifier
                    P, mv_risk = dNDF_mv.predict_MV(Xs_test, y_test)
                    
                    # Compute the bounds and risks for the separate views classifiers
                    if bound not in MV_SPECIFIC_BOUNDS:
                        v_bounds = []
                        for v in range(len(Xs_test)):
                            v_bound, _, _, _, _, _ = dNDF_per_view[v].bound(
                                            bound=bound,
                                            labeled_data=(Xs_train[v], y_train),
                                            unlabeled_data=unlabeled_data[v] if unlabeled_data else None,
                                            incl_oob=False,
                                            alpha=1)
                            v_bounds.append(v_bound)
                        concat_bound, _, _, _, _, _ = dNDF_per_view[-1].bound(
                                            bound=bound,
                                            labeled_data=(X_train_concat, y_train),
                                            unlabeled_data=c_unlabeled_data,
                                            incl_oob=False,
                                            alpha=1)
                        v_bounds.append(concat_bound)
                        v_risks = [dNDF_per_view[v].predict(Xs_test[v], y_test)[1]
                                   for v in range(len(Xs_test))]
                        v_risks.append(dNDF_per_view[-1].predict(X_test_concat, y_test)[1])
                    else:
                        v_bounds = [np.nan for _ in range(len(Xs_test)+1)]
                        v_risks = [np.nan for _ in range(len(Xs_test)+1)]
                    # print(f"{dNDF_mv.posterior_Qv=} {dNDF_mv.posterior_rho=}")
                    

                    # Save the results
                    print(colored(f"Entering save and stats zone-------------------------------", 'green'))
                    views_risks = {f"View{i+1}": v_risks[i] for i in range(len(v_risks)-1)}
                    views_risks.update({"Concatenated": v_risks[-1]})
                    views_risks.update({"Multiview": mv_risk})
                    views_bounds = {f"View{i+1}": v_bounds[i] for i in range(len(v_bounds)-1)}
                    views_bounds.update({"Concatenated": v_bounds[-1]})
                    views_bounds.update({"Multiview": mv_bound})
                    for (kr, r), (kb, b) in zip(views_risks.items(), views_bounds.items()):
                        assert kr == kb # check if the keys are the same
                        exp = {"Run": run+1, 
                            "Bound_name": bound, 
                            "View": kr, 
                            "Risk": "{:.3f}".format(r),
                            "Bound": "{:.3f}".format(b),
                            "R1": "{:.3f}".format(r1),
                            "R2": "{:.3f}".format(r2),
                            "KLqp": "{:.3f}".format(klqp),
                            "KLRpi": "{:.3f}".format(klrpi),
                            "N1": "{:.3f}".format(n1),
                            "N2": "{:.3f}".format(n2),}
                        experiments[s1_size][alpha][config["name"]].append(exp)
                    # TODO: add the posterior_Qv and posterior_rho to the experiment
                # del dNDF_mv, dNDF_per_view
                
            cfg_dir = alpha_dir + "/" + config["name"]
            os.makedirs(cfg_dir, exist_ok=True)
            experiment_df = pd.DataFrame(experiments[s1_size][alpha][config["name"]])
            # example: results/s_labeled-5/alpha-1/stump/MNIST_4vs9_20runs.csv
            file_name = f"{cfg_dir}/{dataset._name}_{label_1}vs{label_2}_{len(RUNS)}runs.csv"
            experiment_df.to_csv(file_name, sep=" ", index=False)