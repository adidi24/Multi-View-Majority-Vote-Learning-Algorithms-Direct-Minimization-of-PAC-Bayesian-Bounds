#
# Implements the TND and DIS bound in theorem 2.3.
#

import numpy as np
from math import ceil, log, sqrt, exp

import torch
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from mvpb.util import uniform_distribution
from mvpb.util import renyi_divergence as rd

def mv_tnd_dis(emp_tnd, emp_dis, nt, nd, RD_QP, RD_rhopi, delta=0.05, lamb1=None, lamb2=None):
    """
    Compute the TND_DIS bound in theorem 2.3.

    Args:
    - emp_tnd (float): The empirical tandem risk.
    - emp_dis (float): The empirical disagreement risk.
    - nt (int): The number of samples for the tandem risk.
    - nd (int): The number of samples for the disagreement.
    - RD_QP (float): the weighted sum of the Rényi divergences between the prior and posterior distributions of each view.
    - RD_rhopi (float): The Rényi divergence between the hyper-prior and hyper-posterior distributions.
    - delta (float, optional): The confidence level. Default is 0.05.

    Returns:
    - float: The TND_DIS bound.
    """
    
    if lamb1 is None or lamb2 is None:
        lamb1 = 2.0 / (sqrt((1.0 * nt * emp_tnd) / (RD_QP + RD_rhopi + log(2.0 * sqrt(nt) / delta)) + 1.0) + 1.0)
        lamb2 = 2.0 / (sqrt((1.0 * nd * emp_dis) / (RD_QP + RD_rhopi + log(2.0 * sqrt(nd) / delta)) + 1.0) + 1.0)
    else:
        lamb1 = lamb1.data.item()
        lamb2 = lamb2.data.item()
    
    loss_term1 = emp_tnd / (1.0 - lamb1 / 2.0) + 2*(RD_QP + RD_rhopi + log((2.0 * sqrt(nt)) / delta)) / (lamb1 * (1.0 - lamb1 / 2.0) * nt)
    loss_term2 = emp_dis / (1.0 - lamb2 / 2.0) + 2*(RD_QP + RD_rhopi + log((2.0 * sqrt(nd)) / delta)) / (lamb2 * (1.0 - lamb2 / 2.0) * nd)

    bound = 2*loss_term1 + loss_term2
    
    return min(1.0, bound)

def tnd_dis(emp_tnd, emp_dis, nt, nd, RD_QP, delta=0.05):
    """
    Compute the TND_DIS bound for each view.

    Args:
    - emp_tnd (float): The empirical tandem risk.
    - emp_dis (float): The empirical disagreement risk.
    - nt (int): The number of samples for the tandem risk.
    - nd (int): The number of samples for the disagreement.
    - RD_QP (float): The Rényi divergence between the prior and posterior distributions.
    - delta (float, optional): The confidence level. Default is 0.05.

    Returns:
    - float: The TND_DIS bound.
    """
    lamb1 = 2.0 / (sqrt((2.0 * nt * emp_tnd) / (2*RD_QP + log(2.0 * sqrt(nt) / delta)) + 1.0) + 1.0)
    lamb2 = 2.0 / (sqrt((2.0 * nd * emp_dis) / (2*RD_QP + log(2.0 * sqrt(nd) / delta)) + 1.0) + 1.0)
    
    loss_term1 = emp_tnd / (1.0 - lamb1 / 2.0) + 2*RD_QP + log((2.0 * sqrt(nt)) / delta) / (lamb1 * (1.0 - lamb1 / 2.0) * nt)
    loss_term2 = emp_dis / (1.0 - lamb2 / 2.0) + 2*RD_QP + log((2.0 * sqrt(nd)) / delta) / (lamb2 * (1.0 - lamb2 / 2.0) * nd)

    bound = 2*loss_term1 + loss_term2
    
    return min(1.0, bound)

def compute_loss(emp_tnd_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, nt, nd, delta, alpha, lamb1=None, lamb2=None):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 4.

     Args:
    - emp_tnd_views (list): A list of empirical tandem risks for each view.
    - emp_dis_views (list): A list of empirical disagreement risks for each view.
    - posterior_Qv (list): A list of posterior distributions for each view.
    - posterior_rho (tensor): The hyper-posterior distribution for the weights.
    - prior_Pv (list): A list of prior distributions for each view.
    - prior_pi (tensor): The hyper-prior distribution for the weights.
    - nt (int): The number of samples for the tandem risk.
    - nd (int): The number of samples for the disagreement.
    - delta (float): The confidence parameter.
    - lamb (float): lambda.

     Returns:
    - tensor: The computed loss value.

     """
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical tandem risk
    emp_tnd_risks = [torch.sum(torch.sum(view * q) * q) for view, q in zip(emp_tnd_views, softmax_posterior_Qv)]
    emp_mv_tnd = torch.sum(torch.sum(torch.stack(emp_tnd_risks) * softmax_posterior_rho) * softmax_posterior_rho)
    
    # Compute the empirical disagreement
    emp_dis_risks = [torch.sum(torch.sum(view * q) * q) for view, q in zip(emp_dis_views, softmax_posterior_Qv)]
    emp_mv_dis = torch.sum(torch.sum(torch.stack(emp_dis_risks) * softmax_posterior_rho) * softmax_posterior_rho)


    # Compute the Rényi divergences
    RD_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
    RD_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    
    if lamb1 is None or lamb2 is None:
        lamb1 = 2.0 / (torch.sqrt((1.0 * nt * emp_mv_tnd) / (RD_QP + RD_rhopi + torch.log(2.0 * torch.sqrt(nt) / delta)) + 1.0) + 1.0)
        lamb2 = 2.0 / (torch.sqrt((1.0 * nd * emp_mv_dis) / (RD_QP + RD_rhopi + torch.log(2.0 * torch.sqrt(nd) / delta)) + 1.0) + 1.0)
    
    loss_term1 = emp_mv_tnd / (1.0 - lamb1 / 2.0) + 2*(RD_QP + RD_rhopi + torch.log((2.0 * torch.sqrt(nt)) / delta)) / (lamb1 * (1.0 - lamb1 / 2.0) * nt)
    loss_term2 = emp_mv_dis / (1.0 - lamb2 / 2.0) + 2*(RD_QP + RD_rhopi + torch.log((2.0 * torch.sqrt(nd)) / delta)) / (lamb2 * (1.0 - lamb2 / 2.0) * nd)

    loss = loss_term1 + loss_term2/2.0
    
    return loss


def optimizeTND_DIS_mv_torch(emp_tnd_views, emp_dis_views, nt, nd, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambdas=False, alpha=1.0):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
    - emp_tnd_views (list): A list of empirical tandem risks for each view.
    - emp_dis_views (list): A list of empirical disagreements for each view.
    - nt (int): The number of samples for the tandem risk.
    - nd (int): The number of samples for the disagreement.
    - delta (float, optional): The confidence level. Default is 0.05.
    - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.

    Returns:
    - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    
    assert len(emp_tnd_views) == len(emp_dis_views)
    m = len(emp_tnd_views[0])
    v = len(emp_tnd_views)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m).to(device)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True).to(device) for k in range(v)])
    for p in prior_Pv:
        p.requires_grad = False

    prior_pi = uniform_distribution(v).to(device)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True).to(device)
    prior_pi.requires_grad = False
    
    for t in emp_tnd_views:
        t = torch.tensor(t).to(device)
    for d in emp_dis_views:
        d = torch.tensor(d).to(device)
        
    
    lamb1, lamb2 = None, None
    if optimise_lambdas:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb1 = torch.nn.Parameter(torch.empty(1).uniform_(0.0001, 1.9999), requires_grad=True)
        lamb2 = torch.nn.Parameter(torch.empty(1).uniform_(0.0001, 1.9999), requires_grad=True)
        all_parameters = list(posterior_Qv) + [posterior_rho] + [lamb1] + [lamb2]
    else:
        all_parameters = list(posterior_Qv) + [posterior_rho] 
    optimizer = optim.SGD(all_parameters, lr=0.01,momentum=0.9)

    prev_loss = float('inf')

    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_loss(emp_tnd_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, nt, nd, delta, alpha, lamb1, lamb2)
    
        loss.backward() # Backpropagation

        torch.nn.utils.clip_grad_norm_(all_parameters, 5.0)
        optimizer.step() # Update the parameters
        
        if optimise_lambdas:
            lamb1.data = lamb1.data.clamp(0.0001, 1.9999)
            lamb2.data = lamb2.data.clamp(0.0001, 1.9999)

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
    return softmax_posterior_Qv, softmax_posterior_rho, lamb1, lamb2
