#
# Implements the DIS bound in theorem 3.3.
#

import numpy as np
from math import ceil, log, sqrt, exp

import torch
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from mvpb.util import uniform_distribution
from mvpb.util import renyi_divergence as rd

def mv_dis(emp_mv_risk, emp_dis, ng, nd, RD_QP, RD_rhopi, delta=0.05, lamb=None, gamma=None):
    """
    Compute the DIS bound in theorem 3.3.

    Args:
    - emp_risk (float): The empirical  risk.
    - emp_dis (float): The empirical disagreement risk.
    - ng (int): The number of samples for the risk.
    - nd (int): The number of samples for the disagreement.
    - RD_QP (float): the weighted sum of the Rényi divergences between the prior and posterior distributions of each view.
    - RD_rhopi (float): The Rényi divergence between the hyper-prior and hyper-posterior distributions.
    - delta (float, optional): The confidence level. Default is 0.05.

    Returns:
    - float: The DIS bound.
    """

    if lamb is None or gamma is None:
        lamb = 2.0 / (sqrt((2.0*ng*emp_mv_risk)/(RD_QP + RD_rhopi + log(4.0 * sqrt(ng) / delta)) + 1.0) + 1.0)
        gamma = sqrt(2.0 * (RD_QP + RD_rhopi + log((4.0*sqrt(nd))/delta)) / (emp_mv_risk*nd))
    else:
        lamb = lamb.data.item()
        gamma = gamma.data.item()
    
    phi = emp_mv_risk / (1.0 - lamb/2.0) + (RD_QP + RD_rhopi + log((4.0*sqrt(ng))/delta))/(lamb*(1.0-lamb/2.0)*ng)
    dis_term = (1-gamma/2.0) * emp_dis - (RD_QP + RD_rhopi + log((4.0*sqrt(nd))/delta))/(gamma*nd)
    
    bound = 4.0*phi - 2.0*dis_term
    
    return min(1.0, bound)

def dis(emp_mv_risk, emp_dis, ng, nd, RD_QP, delta=0.05):
    """
    Compute the DIS bound for each view.

    Args:
    - emp_mv_risk (float): The empirical risk.
    - emp_dis (float): The empirical disagreement risk.
    - ng (int): The number of samples for the risk.
    - nd (int): The number of samples for the disagreement.
    - RD_QP (float): The Rényi divergence between the prior and posterior distributions.
    - delta (float, optional): The confidence level. Default is 0.05.

    Returns:
    - float: The DIS bound.
    """
    
    lamb = 2.0 / (sqrt((2.0*ng*emp_mv_risk)/(RD_QP + log(4.0 * sqrt(ng) / delta)) + 1.0) + 1.0)
    gamma = sqrt(2.0 * (2.0*RD_QP + log((4.0*sqrt(nd))/delta)) / (emp_mv_risk*nd))
    
    phi = emp_mv_risk / (1.0 - lamb/2.0) + (RD_QP + log((4.0*sqrt(ng))/delta))/(lamb*(1.0-lamb/2.0)*ng)
    dis_term = (1-gamma/2.0) * emp_dis - (2.0*RD_QP + log((4.0*sqrt(nd))/delta))/(gamma*nd)
    
    bound = 4.0*phi - 2.0*dis_term
    
    return min(1.0, bound)

def compute_loss(emp_risks_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, alpha, lamb=None, gamma=None):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 4.

     Args:
    - emp_risks_views (list): A list of empirical risks for each view.
    - emp_dis_views (list): A list of empirical disagreement risks for each view.
    - posterior_Qv (list): A list of posterior distributions for each view.
    - posterior_rho (tensor): The hyper-posterior distribution for the weights.
    - prior_Pv (list): A list of prior distributions for each view.
    - prior_pi (tensor): The hyper-prior distribution for the weights.
    - ng (int): The number of samples for the risk.
    - nd (int): The number of samples for the disagreement.
    - delta (float): The confidence parameter.
    - lamb (float): lambda.

     Returns:
    - tensor: The computed loss value.

     """
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical risk
    emp_risks = [torch.sum(view * q) for view, q in zip(emp_risks_views, softmax_posterior_Qv)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks) * softmax_posterior_rho)
    
    # Compute the empirical disagreement
    emp_dis_risks = [torch.sum(torch.sum(view * q) * q) for view, q in zip(emp_dis_views, softmax_posterior_Qv)]
    emp_mv_dis = torch.sum(torch.sum(torch.stack(emp_dis_risks) * softmax_posterior_rho) * softmax_posterior_rho)
    # print(f"{emp_mv_risk=} {emp_mv_dis=}")

    # Compute the Rényi divergences
    RD_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
    RD_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    
    if lamb is None or gamma is None:
        lamb = 2.0 / (torch.sqrt((2.0 * ng * emp_mv_risk) / (RD_QP + RD_rhopi + torch.log(4.0 * torch.sqrt(ng) / delta)) + 1.0) + 1.0)
        gamma = torch.sqrt(2.0 * (RD_QP + RD_rhopi + torch.log((4.0*torch.sqrt(nd))/delta)) / (emp_mv_risk*nd))
    
    phi = emp_mv_risk / (1.0 - lamb / 2.0) + 2*(RD_QP + RD_rhopi + torch.log((4.0 * torch.sqrt(ng)) / delta)) / (lamb * (1.0 - lamb / 2.0) * ng)
    dis_term = (1-gamma/2.0) * emp_mv_dis - (RD_QP + RD_rhopi + torch.log((4.0 * torch.sqrt(nd)) / delta)) / (gamma*nd)

    loss = 2.0*phi - dis_term
    
    return loss


def optimizeDIS_mv_torch(emp_risks_views, emp_dis_views, ng, nd, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambda_gamma=False, alpha=1.0):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
    - emp_risks_views (list): A list of empirical risks for each view.
    - emp_dis_views (list): A list of empirical disagreements for each view.
    - ng (int): The number of samples for the risk.
    - nd (int): The number of samples for the disagreement.
    - delta (float, optional): The confidence level. Default is 0.05.
    - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.

    Returns:
    - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    
    assert len(emp_risks_views) == len(emp_dis_views)
    m = len(emp_dis_views[0])
    v = len(emp_dis_views)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m).to(device)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True).to(device) for k in range(v)])
    for p in prior_Pv:
        p.requires_grad = False

    prior_pi = uniform_distribution(v).to(device)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True).to(device)
    prior_pi.requires_grad = False
    
    emp_risks_views = torch.tensor(emp_risks_views).to(device)
    emp_dis_views = torch.tensor(emp_dis_views).to(device)
    
    lamb, gamma = None, None
    if optimise_lambda_gamma:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb = torch.nn.Parameter(torch.empty(1).uniform_(0.0001, 1.9999), requires_grad=True).to(device)
        gamma = torch.nn.Parameter(torch.empty(1).uniform_(0.0001), requires_grad=True).to(device)
        all_parameters = list(posterior_Qv) + [posterior_rho] + [lamb] + [gamma]
    else:
        all_parameters = list(posterior_Qv) + [posterior_rho] 
    optimizer = optim.SGD(all_parameters, lr=0.01,momentum=0.9)

    prev_loss = float('inf')

    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_loss(emp_risks_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, alpha, lamb, gamma)
    
        loss.backward() # Backpropagation
    
        torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        if optimise_lambda_gamma:
            lamb.data = lamb.data.clamp(0.0001, 1.9999)
            gamma.data = gamma.data.clamp_min(0.0001)
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
    return softmax_posterior_Qv, softmax_posterior_rho, lamb, gamma
