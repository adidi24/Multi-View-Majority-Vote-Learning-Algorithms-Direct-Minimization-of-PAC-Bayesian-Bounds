#
# Implementation of the lambda bound and optimization procedure.
#
# Based on paper:
# [Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin.
#  A strongly quasiconvex385PAC-Bayesian bound. InAlgorithmic Learning Theory (ALT), 2017] 
#

import torch
import torch.nn.functional as F
import torch.optim as optim


import numpy as np

from math import ceil, log, sqrt, exp
from util import kl, uniform_distribution


# Compute PAC-Bayes-Lambda-bound:
def lamb(emp_risk, n, KL_qp, delta=0.05):
    n = float(n)

    lamb = 2.0 / (sqrt((2.0*n*emp_risk)/(KL_qp+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
    bound = emp_risk / (1.0 - lamb/2.0) + (KL_qp + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)

    return min(1.0,2.0*bound)

# Compute MV-PAC-Bayes-Lambda-bound:
def mv_lamb(emp_mv_risk, n, KL_qp, KL_rhopi, delta=0.05):
    n = float(n)

    lamb = 2.0 / (sqrt((2.0*n*emp_mv_risk)/(KL_qp+KL_rhopi+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
    bound = emp_mv_risk / (1.0 - lamb/2.0) + (KL_qp + KL_rhopi + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)

    return min(1.0,2.0*bound)

# Optimize PAC-Bayes-Lambda-bound:
def optimizeLamb(emp_risks, n, delta=0.05, eps=10**-9, abc_pi=None):
    m = len(emp_risks)
    n = float(n)
    prior  = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    posterior = uniform_distribution(m) if abc_pi is None else np.copy(abc_pi)
    KL_qp = kl(posterior,prior)

    lamb = 1.0
    emp_risk = np.average(emp_risks, weights=posterior)

    upd = emp_risk / (1.0 - lamb/2.0) + (KL_qp + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)
    bound = upd+2*eps

    while bound-upd > eps:
        bound = upd
        lamb = 2.0 / (sqrt((2.0*n*emp_risk)/(KL_qp+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
        for h in range(m):
            posterior[h] = prior[h]*exp(-lamb*n*emp_risks[h])
        posterior /= np.sum(posterior)

        emp_risk = np.average(emp_risks, weights=posterior)
        KL_qp = kl(posterior,prior)

        upd = emp_risk / (1.0 - lamb/2.0) + (KL_qp + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n) 
    return bound, posterior, lamb, KL_qp, emp_risk


# Optimize PAC-Bayes-Multi-View-Lambda-bound:
def optimizeLamb_mv(emp_risks, KL_qps, n, delta=0.05, eps=10**-9):
    v = len(KL_qps)
    # emp_risks  = np.array([0.10602932, 0.1056597 , 0.10573632, 0.10610876])
    # print(f"{emp_risks=}")
    n = float(n)
    pi  = uniform_distribution(v)
    rho = uniform_distribution(v)
    KL_rhopi = kl(rho,pi) #KL of hyper distributions
    KL_qp = np.average(KL_qps, weights=rho)
    
    lamb = 1.0
    # print(f"{rho=}")
    emp_mv_risk = np.average(emp_risks, weights=rho, axis=0)
    # print(f"{emp_mv_risk=}")
    upd = emp_mv_risk / (1.0 - lamb/2.0) + (KL_qp + KL_rhopi + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)
    bound = upd+2*eps

    while bound-upd > eps:
        # print(f"{upd=}\n {rho=}\n {emp_mv_risk}")
        bound = upd
        lamb = 2.0 / (sqrt((2.0*n*emp_mv_risk)/(KL_qp+KL_rhopi+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
        for h in range(v):
            rho[h] = pi[h]*exp(-lamb*n*emp_risks[h])
        rho /= np.sum(rho)

        emp_mv_risk = np.average(emp_risks, weights=rho, axis=0)
        KL_rhopi = kl(rho,pi)
        KL_qp = np.average(KL_qps, weights=rho)
        
        upd = emp_mv_risk / (1.0 - lamb/2.0) + (KL_qp + KL_rhopi + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)
    return bound, rho, lamb, KL_rhopi, emp_mv_risk


def mv_lamb(emp_risks_views, n, delta=0.05):
    n = float(n)
    
    emp_risks_views = np.average(emp_risks_views, weights=posterior)
    lamb = 2.0 / (sqrt((2.0*n*emp_mv_risk)/(KL_qp+KL_rhopi+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
    bound = emp_mv_risk / (1.0 - lamb/2.0) + (KL_qp + KL_rhopi + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)

    return min(1.0,2.0*bound)

def compute_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, lamb):
    # Appliquer softmax pour normaliser posterior_Qv et posterior_rho
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Calculer les risques empiriques en utilisant les versions normalisées
    emp_risks = [torch.sum(torch.tensor(view) * q) for view, q in zip(emp_risks_views, softmax_posterior_Qv)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks) * softmax_posterior_rho)

    # Calculer KL divergence en utilisant les versions normalisées
    KL_QP = torch.sum(torch.stack([kl(q, p) * softmax_posterior_rho for q, p in zip(softmax_posterior_Qv, prior_Pv)]))
    KL_rhopi = kl(softmax_posterior_rho, prior_pi)
    lamb = 2.0 / (torch.sqrt((2.0 * n * emp_mv_risk) / (KL_QP + KL_rhopi + torch.log(2.0 * torch.sqrt(n) / delta)) + 1.0) + 1.0)


    # Calculer la perte totale
    loss = emp_mv_risk / (1.0 - lamb / 2.0) + (KL_QP + KL_rhopi + torch.log((2.0 * torch.sqrt(n)) / delta)) / (lamb * (1.0 - lamb / 2.0) * n)

    return loss

# def optimizeLamb_mv_torch(emp_risks_views, ns_min_values, delta=0.05, eps=10**-9,lambda_sum = 1,lambda_l2 = 0):
#     m = len(emp_risks_views[0])
#     v = len(emp_risks_views)
#     n = torch.tensor(np.min(ns_min_values))
#     print(emp_risks_views)
#     # Initialisation des distributions
#     prior_Pv = [uniform_distribution(m) for k in range(v)]
#     posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True) for k in range(v)])

#     prior_pi = uniform_distribution(v)
#     posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True)
    
#     lamb = 1.0
#     # print(f"{rho=}")`
#     emp_risks = [torch.sum(torch.tensor(view) * posterior_Qv[i]) for i, view in enumerate(emp_risks_views)]
#     emp_mv_risk = torch.sum(torch.stack(emp_risks)*posterior_rho)
#     # print(f"{emp_mv_risk=}")
#     KL_QP = torch.sum(torch.stack([kl(posterior_Qv[k], prior_Pv[k]) * posterior_rho for k in range(v)]))

#     KL_rhopi = kl(posterior_rho,prior_pi)

#     upd = emp_mv_risk / (1.0 - lamb/2.0) + (KL_QP + KL_rhopi + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)
#     bound = upd+2*eps
#     print("Bound:", bound)
#     print("Upd:", upd)
#     print("Différence:", bound - upd)
    
#     all_parameters = list(posterior_Qv) + [posterior_rho]
#     optimizer = optim.SGD(all_parameters, lr=0.01,momentum=0.9)

    
#     for i in range(100):
        
        
#         optimizer.zero_grad()
#         # print('test',F.softmax(posterior_Qv[0], dim=0))
#         print(posterior_Qv[0])
#         print(posterior_rho)
#         # Recalculer emp_risks, emp_mv_risk, KL_QP, KL_rhopi à partir des valeurs actuelles de posterior_Qv et posterior_rho
#         emp_risks = [torch.sum(torch.tensor(view) * posterior_Qv[i]) for i, view in enumerate(emp_risks_views)]
#         emp_mv_risk = torch.sum(torch.stack(emp_risks) * posterior_rho)
#         KL_QP = torch.sum(torch.stack([kl(posterior_Qv[k], prior_Pv[k]) * posterior_rho for k in range(v)]))
#         KL_rhopi = kl(posterior_rho, prior_pi)
#         # l2_penalty = torch.sum(torch.stack([torch.norm(q, p=2) for q in posterior_Qv]))
#         # sum_penalty = torch.sum(torch.stack([(q.sum() - 1)**2 for q in posterior_Qv]))
        
#         # Recalculer upd et bound en utilisant les nouvelles valeurs de emp_mv_risk, KL_QP, KL_rhopi
#         upd = emp_mv_risk / (1.0 - lamb / 2.0) + (KL_QP + KL_rhopi + torch.log((2.0 * torch.sqrt(n)) / delta)) / (lamb * (1.0 - lamb / 2.0) * n)
#         # upd += lambda_l2*l2_penalty 
        
#         # Mise à jour de lamb si nécessaire
#         lamb = 2.0 / (torch.sqrt((2.0 * n * emp_mv_risk) / (KL_QP + KL_rhopi + torch.log(2.0 * torch.sqrt(n) / delta)) + 1.0) + 1.0)
#         # Nouveau calcul de bound pour la condition de la boucle
#         bound = emp_mv_risk / (1.0 - lamb / 2.0) + (KL_QP + KL_rhopi + torch.log((2.0 * torch.sqrt(n)) / delta)) / (lamb * (1.0 - lamb / 2.0) * n)
#         # l2_penalty = torch.sum(torch.stack([torch.norm(q, p=2) for q in posterior_Qv]))
#         # sum_penalty = torch.sum(torch.stack([(q.sum() - 1)**2 for q in posterior_Qv]))
        
#         # bound += lambda_l2*l2_penalty

#         # Rétropropagation et mise à jour des paramètres
#         bound.backward()
#         optimizer.step()
#         with torch.no_grad():
#             for q in posterior_Qv:
#                 q.clamp_(1/(m*10),1-1/(m*10))
#                 q.div_(torch.norm(F.relu(q), p=1))
#                 # q[:] = F.softmax(q, dim=0)
                
                
#             posterior_rho.clamp_(1/(v*10),1-1/(v*10))
#             posterior_rho.div_(torch.norm(F.relu(posterior_rho), p=1))
#             # posterior_rho[:] = F.softmax(posterior_rho, dim=0)
            
#         # Condition de sortie
#         # if torch.abs(bound - upd) <= eps:
#         #     break                
            
    
#     # print("Bound après mise à jour:", bound)
#     # print("Upd après mise à jour:", upd)
#     # print("Différence:", abs(bound - upd))
        
#     # posterior_Qv = [F.softmax(param, dim=0) for param in posterior_Qv]
#     # posterior_rho = F.softmax(posterior_rho, dim=0)


#     return posterior_Qv, posterior_rho

def optimizeLamb_mv_torch(emp_risks_views, ns_min_values, delta=0.05, eps=10**-9,lambda_sum = 1,lambda_l2 = 0):
    m = len(emp_risks_views[0])
    v = len(emp_risks_views)
    n = torch.tensor(np.min(ns_min_values))
    print(emp_risks_views)
    # Initialisation des distributions
    prior_Pv = [uniform_distribution(m) for k in range(v)]
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True) for k in range(v)])

    prior_pi = uniform_distribution(v)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True)
    
    lamb = 1.0
    # print(f"{rho=}")`
    emp_risks = [torch.sum(torch.tensor(view) * posterior_Qv[i]) for i, view in enumerate(emp_risks_views)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks)*posterior_rho)
    # print(f"{emp_mv_risk=}")
    KL_QP = torch.sum(torch.stack([kl(posterior_Qv[k], prior_Pv[k]) * posterior_rho for k in range(v)]))

    KL_rhopi = kl(posterior_rho,prior_pi)

    upd = emp_mv_risk / (1.0 - lamb/2.0) + (KL_QP + KL_rhopi + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)
    bound = upd+2*eps
    print("Bound:", bound)
    print("Upd:", upd)
    print("Différence:", bound - upd)
    
    all_parameters = list(posterior_Qv) + [posterior_rho]
    optimizer = optim.SGD(all_parameters, lr=0.01,momentum=0.9)

    
    prev_loss = float('inf')

    # Dans votre boucle d'optimisation
    for i in range(100):
        optimizer.zero_grad()
    
        # Calculer la perte
        loss = compute_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, lamb)
    
        # Rétropropagation
        loss.backward()
    
        # Mise à jour des paramètres
        optimizer.step()
    
        # Vérifier la condition de sortie basée sur le changement de la perte
        if torch.abs(prev_loss - loss).item() <= eps:
            break
    
        prev_loss = loss.item()  # Mettre à jour prev_loss avec la valeur actuelle de la perte
    
        # Optionnel: Afficher la perte pour le suivi
        print(f"Iteration {i}, Loss: {loss.item()}")

    # Après la boucle d'optimisation
    with torch.no_grad():
        softmax_posterior_Qv = [torch.softmax(q, dim=0) for q in posterior_Qv]
        softmax_posterior_rho = torch.softmax(posterior_rho, dim=0)
    return posterior_Qv, posterior_rho

# def optimizeLamb_mv_torch(emp_risks_views, ns_min_values, delta=0.05, eps=10**-9, lambda_sum=1, lambda_l2=0):
#     m = len(emp_risks_views[0])
#     v = len(emp_risks_views)
#     n = torch.tensor(np.min(ns_min_values), dtype=torch.float64)
    
#     # Initialisation des distributions
#     prior_Pv = [uniform_distribution(m) for k in range(v)]
#     posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True) for k in range(v)])
#     prior_pi = uniform_distribution(v)
#     posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True)
#     emp_risks = [torch.sum(torch.tensor(view, dtype=torch.float64) * posterior_Qv[i]) for i, view in enumerate(emp_risks_views)]
#     emp_mv_risk = torch.sum(torch.stack(emp_risks) * posterior_rho)
#     KL_QP = torch.sum(torch.stack([kl(posterior_Qv[k], prior_Pv[k]) * posterior_rho for k in range(v)]))
#     KL_rhopi = kl(posterior_rho, prior_pi)
    
#     # Regroupement de tous les paramètres pour l'optimiseur
#     all_parameters = list(posterior_Qv) + [posterior_rho]
#     optimizer = optim.LBFGS(all_parameters, lr=0.1,max_iter=1)


#     def closure():
#         optimizer.zero_grad()
#         print(posterior_Qv[0])
#         print(posterior_rho)
#         # Votre calcul de la fonction de perte ici...
#         emp_risks = [torch.sum(torch.tensor(view, dtype=torch.float64) * posterior_Qv[i]) for i, view in enumerate(emp_risks_views)]
#         emp_mv_risk = torch.sum(torch.stack(emp_risks) * posterior_rho)
#         KL_QP = torch.sum(torch.stack([kl(posterior_Qv[k], prior_Pv[k]) * posterior_rho for k in range(v)]))
#         KL_rhopi = kl(posterior_rho, prior_pi)
#         # Mise à jour de lamb si nécessaire
#         lamb = 2.0 / (torch.sqrt((2.0 * n * emp_mv_risk) / (KL_QP + KL_rhopi + torch.log(2.0 * torch.sqrt(n) / delta)) + 1.0) + 1.0)
#         bound = emp_mv_risk / (1.0 - lamb / 2.0) + (KL_QP + KL_rhopi + torch.log((2.0 * torch.sqrt(n)) / delta)) / (lamb * (1.0 - lamb / 2.0) * n)
        
#         # Rétropropagation
#         bound.backward()
#         return bound

#     for i in range(100):
#         # Optimisation avec LBFGS
#         optimizer.step(closure)
        
#         with torch.no_grad():
#             # Application de softmax pour s'assurer que les poids sont des distributions de probabilités
#             for k, q in enumerate(posterior_Qv):
#                 print("Avant normalisation et clampage:", q)
#                 posterior_Qv[k] = torch.nn.Parameter(torch.softmax(q, dim=0))
#                 print("Après normalisation et clampage:", q)
            
#             posterior_rho.data = torch.softmax(posterior_rho, dim=0)  

#     return posterior_Qv, posterior_rho

