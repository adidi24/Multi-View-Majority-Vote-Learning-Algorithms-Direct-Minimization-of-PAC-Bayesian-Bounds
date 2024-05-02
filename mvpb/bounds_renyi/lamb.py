#
# Implementation of the lambda bound and optimization procedure.
#
# Based on paper:
# [Nirdas Thiemann, Christian Igel, Olivier Wintenberger, and Yevgeny Seldin.
#  A strongly quasiconvex385PAC-Bayesian bound. InAlgorithmic Learning Theory (ALT), 2017] 
#

import numpy as np
from math import ceil, log, sqrt, exp

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..cocob_optim import COCOB

from mvpb.tools import solve_kl_sup
from mvpb.util import uniform_distribution
from mvpb.util import renyi_divergence as rd
from mvpb.util import LogBarrierFunction as lbf

def PBkl(empirical_gibbs_risk, m, RD_QP, delta=0.05):
    """ PAC Bound ZERO of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

    Compute a PAC-Bayesian upper bound on the Bayes risk by
    multiplying by two an upper bound on the Gibbs risk

    empirical_gibbs_risk : Gibbs risk on the training set
    m : number of training examples
    KL_qp : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    # Don't validate - gibbs_risk may be > 0.5 in non-binary case 
    #if not validate_inputs(empirical_gibbs_risk, None, m, KL_qp, delta): return 1.0

    xi_m = 2*sqrt(m)
    right_hand_side = ( RD_QP + log( xi_m / delta ) ) / m
    sup_R = min(1.0, solve_kl_sup(empirical_gibbs_risk, right_hand_side))

    return 2 * sup_R

def PBkl_MV(empirical_gibbs_risk, m, RD_qp, RD_rhopi, delta=0.05):
    """ Multi view PAC Bound ZERO
    """

    xi_m = 2*sqrt(m)
    right_hand_side = (RD_qp + RD_rhopi + log( xi_m / delta ) ) / m
    sup_R = min(0.5, solve_kl_sup(empirical_gibbs_risk, right_hand_side))

    return 2 * sup_R

def compute_mv_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, lamb=None, alpha=1.1):
    """
    Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 2.

    Args:
        - emp_risks_views (list): A list of empirical risks for each view.
        - posterior_Qv (list): A list of posterior distributions for each view.
        - posterior_rho (tensor): The hyper-posterior distribution for the weights.
        - prior_Pv (list): A list of prior distributions for each view.
        - prior_pi (tensor): The hyper-prior distribution for the weights.
        - n (int): The number of samples.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.

    Returns:
        -  tensor: The computed loss value.

    """
    
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical risk
    emp_risks = [torch.sum(view * q) for view, q in zip(emp_risks_views, softmax_posterior_Qv)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks) * softmax_posterior_rho)

    # Compute the Rényi divergences
    RD_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
    RD_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    
    if lamb is None:
        lamb = 2.0 / (torch.sqrt((2.0 * n * emp_mv_risk) / (RD_QP + RD_rhopi + torch.log(2.0 * torch.sqrt(n) / delta)) + 1.0) + 1.0)

    loss = emp_mv_risk / (1.0 - lamb / 2.0) + (RD_QP + RD_rhopi + torch.log((2.0 * torch.sqrt(n)) / delta)) / (lamb * (1.0 - lamb / 2.0) * n)
    
    return 2.0*loss, loss


def optimizeLamb_mv_torch(emp_risks_views, n, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambda=False, alpha=1.1, t=100):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks_views (list): A list of empirical risks for each view.
        - n (list): The number of samples.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.

    Returns:
        - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    log_barrier = lbf(t)
    m = len(emp_risks_views[0])
    v = len(emp_risks_views)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m).to(device)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True).to(device) for k in range(v)])
    for p in prior_Pv:
        p.requires_grad = False
        

    prior_pi = uniform_distribution(v).to(device)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True).to(device)
    prior_pi.requires_grad = False
    
    emp_risks_views = torch.tensor(emp_risks_views).to(device)
    
    lamb = None
    if optimise_lambda:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor, 0.0001, 1.9999)
        lamb = torch.nn.Parameter(lamb_tensor)
        
        all_parameters = list(posterior_Qv) + [posterior_rho, lamb]
    else:
        all_parameters = list(posterior_Qv) + [posterior_rho] 
        
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')

    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint = compute_mv_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, lamb, alpha)
        loss = loss + log_barrier(constraint-0,5)
        loss.backward() # Backpropagation
    
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        if optimise_lambda:
            lamb.data = lamb.data.clamp(0.0001, 1.9999)
        # Verify the convergence criteria of the loss
        if torch.abs(prev_loss - loss).item() <= eps:
            print(f"\t Convergence reached after {i} iterations")
            break
    
        prev_loss = loss.item()  # Update the previous loss with the current loss
        # Optional: Display the loss for monitoring
        # print(f"Iteration: {i},\t Loss: {loss.item()}")

    # After the optimization
    with torch.no_grad():
        softmax_posterior_Qv = [torch.softmax(q, dim=0) for q in posterior_Qv]
        softmax_posterior_rho = torch.softmax(posterior_rho, dim=0)
    return softmax_posterior_Qv, softmax_posterior_rho, lamb



def compute_loss(emp_risks, posterior_Q, prior_P, n, delta, lamb=None, gamma=None, alpha=1.1):
    """
     Compute the loss function for the Majority Vote Learning algorithm.

     Args:
        - emp_risks (float): The empirical risk for a view.
        - posterior_Q (torch.nn.Parameter): The posterior distribution for a view.
        - prior_P (torch.nn.Parameter): The prior distributions for a view.
        - n (int): The number of samples for the risk.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.
        - t (float, optional): Controls the steepness and sensitivity of the barrier. Higher values make the barrier more aggressive. Default is 100.


     Returns:
        - tensor: The computed loss value.

     """
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Q = F.softmax(posterior_Q, dim=0)
    
    # Compute the empirical risk
    emp_risk = torch.sum(emp_risks * softmax_posterior_Q)

    # Compute the Rényi divergence
    RD_QP = rd(softmax_posterior_Q, prior_P, alpha)
    
    if lamb is None or gamma is None:
        lamb = 2.0 / (torch.sqrt((2.0 * n * emp_risk) / (RD_QP + torch.log(2.0 * torch.sqrt(n) / delta)) + 1.0) + 1.0)

    loss = emp_risk / (1.0 - lamb / 2.0) + (RD_QP +  torch.log((2.0 * torch.sqrt(n)) / delta)) / (lamb * (1.0 - lamb / 2.0) * n)
    
    return 2.0*loss


def optimizeLamb_torch(emp_risks, n, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambda=False, alpha=1.1):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks (float): The empirical risk for a view.
        - n (int): The number of samples for the risk.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.

    Returns:
        - tuple: A tuple containing the optimized posterior distribution for a view (posterior_Q).
    """
    
    m = len(emp_risks)
    
    # Initialisation with the uniform distribution
    prior_P = uniform_distribution(m).to(device)
    posterior_Q = torch.nn.Parameter(prior_P.clone(), requires_grad=True).to(device)
    # We don't need to compute the gradients for the  hyper-prior too
    prior_P.requires_grad = False
    
    # Convert the empirical risks and disagreements to tensors
    emp_risks = torch.tensor(emp_risks).to(device)
    
    lamb = None
    if optimise_lambda:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor, 0.0001, 1.9999)
        lamb = torch.nn.Parameter(lamb_tensor)
        
        all_parameters = [posterior_Q, lamb]
    else:
        all_parameters = [posterior_Q]
        
    # Optimizer
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_loss(emp_risks, posterior_Q, prior_P, n, delta, lamb, alpha)
    
        loss.backward() # Backpropagation
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        if optimise_lambda:
            # Clamping the values of lambda
            lamb.data = lamb.data.clamp(0.0001, 1.9999)
        # Verify the convergence criteria of the loss
        if torch.abs(prev_loss - loss).item() <= eps:
            print(f"\t Convergence reached after {i} iterations")
            break
    
        prev_loss = loss.item()  # Update the previous loss with the current loss
        # Optional: Display the loss for monitoring
        # print(f"Iteration: {i},\t Loss: {loss.item()}")

    # After the optimization: Apply the softmax to the posterior distribution 
    # to ensure that the weights are probability distributions
    with torch.no_grad():
        softmax_posterior_Q = torch.softmax(posterior_Q, dim=0)
    return softmax_posterior_Q, lamb
