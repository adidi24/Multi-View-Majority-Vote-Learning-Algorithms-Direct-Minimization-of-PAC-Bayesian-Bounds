#
# Implementation of the lambda bound and optimization procedure.
#
# Based on paper:
# [Niklas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin.
#  A strongly quasiconvex385PAC-Bayesian bound. InAlgorithmic Learning Theory (ALT), 2017] 
#

import numpy as np
from math import ceil, log, sqrt, exp

import torch
import torch.nn.functional as F
import torch.optim as optim

from mvpb.util import uniform_distribution
from mvpb.util import renyi_divergence as rd


# PAC-Bayes-Lambda-bound:
def lamb(emp_risk, n, RD_QP, delta=0.05):
    """
    Calculate the value of lambda and the corresponding bound.

    Parameters:
    - emp_risk (float): The empirical risk.
    - n (int): The number of samples.
    - RD_QP (float): The Rényi divergence between the prior and posterior distributions.
    - delta (float, optional): The confidence level. Default is 0.05.

    Returns:
    - float: The minimum of 1.0 and 2.0 times the calculated bound.

    """
    n = float(n)

    lamb = 2.0 / (sqrt((2.0*n*emp_risk)/(RD_QP+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
    bound = emp_risk / (1.0 - lamb/2.0) + (RD_QP + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)

    return min(1.0,2.0*bound)

# MV-PAC-Bayes-Lambda-bound:
def mv_lamb(emp_mv_risk, n, RD_QP, RD_rhopi, delta=0.05, lamb=None):
    """
    Calculate the value of lambda and the corresponding multiview bound.

    Parameters:
    - emp_risk (float): The weighted sum of the empirical risk of each view.
    - n (int): The number of samples.
    - RD_QP (float): the weighted sum of the Rényi divergences between the prior and posterior distributions of each view.
    - RD_rhopi (float): The Rényi divergence between the hyper-prior and hyper-posterior distributions.
    - delta (float, optional): The confidence level. Default is 0.05.

    Returns:
    - float: The calculated PB-lambda bound.

    """
    n = float(n)

    if lamb is None:
        lamb = 2.0 / (sqrt((2.0*n*emp_mv_risk)/(RD_QP+RD_rhopi+log(2.0*sqrt(n)/delta)) + 1.0) + 1.0)
    else:
        lamb = lamb.data.item()
    bound = emp_mv_risk / (1.0 - lamb/2.0) + (RD_QP + RD_rhopi + log((2.0*sqrt(n))/delta))/(lamb*(1.0-lamb/2.0)*n)

    return min(1.0,2.0*bound)

def compute_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, alpha, lamb=None):
    """
    Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 2.

    Args:
        emp_risks_views (list): A list of empirical risks for each view.
        posterior_Qv (list): A list of posterior distributions for each view.
        posterior_rho (tensor): The hyper-posterior distribution for the weights.
        prior_Pv (list): A list of prior distributions for each view.
        prior_pi (tensor): The hyper-prior distribution for the weights.
        n (int): The number of samples.
        delta (float): The confidence parameter.
        lamb (float): lambda.

    Returns:
        tensor: The computed loss value.

    """
    
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical risk
    emp_risks = [torch.sum(torch.tensor(view) * q) for view, q in zip(emp_risks_views, softmax_posterior_Qv)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks) * softmax_posterior_rho)

    # Compute the Rényi divergences
    RD_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
    RD_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    
    if lamb is None:
        lamb = 2.0 / (torch.sqrt((2.0 * n * emp_mv_risk) / (RD_QP + RD_rhopi + torch.log(2.0 * torch.sqrt(n) / delta)) + 1.0) + 1.0)

    loss = emp_mv_risk / (1.0 - lamb / 2.0) + (RD_QP + RD_rhopi + torch.log((2.0 * torch.sqrt(n)) / delta)) / (lamb * (1.0 - lamb / 2.0) * n)
    
    return loss


def optimizeLamb_mv_torch(emp_risks_views, n, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambda=False, alpha=1.0):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        emp_risks_views (list): A list of empirical risks for each view.
        n (list): The number of samples.
        delta (float, optional): The confidence level. Default is 0.05.
        eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.

    Returns:
        tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    
    m = len(emp_risks_views[0])
    v = len(emp_risks_views)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True) for k in range(v)])

    prior_pi = uniform_distribution(v)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True)
    
    lamb = None
    if optimise_lambda:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb = torch.nn.Parameter(torch.empty(1).uniform_(0.0001, 1.9999), requires_grad=True)
        all_parameters = list(posterior_Qv) + [posterior_rho] + [lamb]
    else:
        all_parameters = list(posterior_Qv) + [posterior_rho] 
        
    optimizer = optim.SGD(all_parameters, lr=0.05,momentum=0.9)

    prev_loss = float('inf')

    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, alpha, lamb)
    
        loss.backward() # Backpropagation
    
    
        torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        if optimise_lambda:
            lamb.data = lamb.data.clamp(0.0001, 1.9999)
        # Verify the convergence criteria of the loss
        if torch.abs(prev_loss - loss).item() <= eps:
            print(f"\t Convergence reached after {i} iterations")
            break
    
        prev_loss = loss.item()  # Update the previous loss with the current loss
        # Optionnel: Afficher la perte pour le suivi
        # print(f"Iteration: {i},\t Loss: {loss.item()}")

    # After the optimization
    with torch.no_grad():
        softmax_posterior_Qv = [torch.softmax(q, dim=0) for q in posterior_Qv]
        softmax_posterior_rho = torch.softmax(posterior_rho, dim=0)
    return softmax_posterior_Qv, softmax_posterior_rho, lamb