#
# Implements the TND and DIS bound in theorem 2.3.
#

import numpy as np
from math import ceil, log, sqrt, exp

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..bounds.cocob_optim import COCOB

from mvpb.bounds.tools import solve_kl_inf, solve_kl_sup
from mvpb.util import kl, uniform_distribution

def TND_DIS(tandem_risk, disagreement, nt, nd, KL_QP, delta=0.05):
    t_rhs = ( 2.0*KL_QP + log(4.0*sqrt(nt)/delta) ) / nt
    t_ub  = min(1.0, solve_kl_sup(tandem_risk, t_rhs))
    
    d_rhs = ( 2.0*KL_QP + log(4.0*sqrt(nd)/delta) ) / nd
    d_lb  = solve_kl_sup(disagreement, d_rhs)
    return min(1.0, 2*t_ub + d_lb)

# Implementation of DIS
def TND_DIS_MV(tandem_risk, disagreement, nt, nd, KL_QP, KL_rhopi, delta=0.05):
    t_rhs = ( 2.0*(KL_QP +  KL_rhopi) + log(4.0*sqrt(nt)/delta) ) / nt
    t_ub  = min(1.0, solve_kl_sup(tandem_risk, t_rhs))
    
    d_rhs = ( 2.0*(KL_QP + KL_rhopi) + log(4.0*sqrt(nd)/delta) ) / nd
    d_lb  = solve_kl_sup(disagreement, d_rhs)
    return min(1.0, 2*t_ub +  d_lb)

def compute_mv_loss(emp_tnd_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, nt, nd, delta, lamb1=None, lamb2=None):
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
    nb_views = len(emp_tnd_views)
     
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical tandem risk
    emp_tnd_v = torch.zeros((nb_views, nb_views))
    for i in range(nb_views):
        for j in range(nb_views):
            print(f"{emp_tnd_views[i, j]=}")
            emp_tnd_v[i, j] = torch.sum(torch.sum(emp_tnd_views[i, j]*softmax_posterior_Qv[i], dim=0) * softmax_posterior_Qv[j])
    emp_mv_tnd =  torch.sum(torch.sum(emp_tnd_v*softmax_posterior_rho, dim=0) * softmax_posterior_rho)
    # emp_tnd_risks = [torch.sum(torch.sum(view * q) * q) for view, q in zip(emp_tnd_views, softmax_posterior_Qv)]
    # emp_mv_tnd = torch.sum(torch.sum(torch.stack(emp_tnd_risks) * softmax_posterior_rho) * softmax_posterior_rho)
    
    # Compute the empirical disagreement
    emp_dis_v = torch.zeros((nb_views, nb_views))
    for i in range(nb_views):
        for j in range(nb_views):
            emp_dis_v[i, j] = torch.sum(torch.sum(emp_dis_views[i, j]*softmax_posterior_Qv[i], dim=0) * softmax_posterior_Qv[j])
    emp_mv_dis =  torch.sum(torch.sum(emp_dis_v*softmax_posterior_rho, dim=0) * softmax_posterior_rho)
    # emp_dis_risks = [torch.sum(torch.sum(view * q) * q) for view, q in zip(emp_dis_views, softmax_posterior_Qv)]
    # emp_mv_dis = torch.sum(torch.sum(torch.stack(emp_dis_risks) * softmax_posterior_rho) * softmax_posterior_rho)


    # Compute the Kullback-Leibler divergences
    KL_QP = torch.sum(torch.stack([kl(q, p)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
    KL_rhopi = kl(softmax_posterior_rho, prior_pi)
    
    if lamb1 is None or lamb2 is None:
        lamb1 = 2.0 / (torch.sqrt((1.0 * nt * emp_mv_tnd) / (KL_QP + KL_rhopi + torch.log(2.0 * torch.sqrt(nt) / delta)) + 1.0) + 1.0)
        lamb2 = 2.0 / (torch.sqrt((1.0 * nd * emp_mv_dis) / (KL_QP + KL_rhopi + torch.log(2.0 * torch.sqrt(nd) / delta)) + 1.0) + 1.0)
    
    loss_term1 = emp_mv_tnd / (1.0 - lamb1 / 2.0) + (2*(KL_QP + KL_rhopi) + torch.log((4.0 * torch.sqrt(nt)) / delta)) / (lamb1 * (1.0 - lamb1 / 2.0) * nt)
    loss_term2 = emp_mv_dis / (1.0 - lamb2 / 2.0) + (2*(KL_QP + KL_rhopi) + torch.log((4.0 * torch.sqrt(nd)) / delta)) / (lamb2 * (1.0 - lamb2 / 2.0) * nd)

    loss = 2.0*loss_term1 + loss_term2
    
    return loss


def optimizeTND_DIS_mv_torch(emp_tnd_views, emp_dis_views, nt, nd, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambdas=False):
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
    m = len(emp_tnd_views[0, 0])
    v = len(emp_tnd_views)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m).to(device)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True).to(device) for k in range(v)])
    for p in prior_Pv:
        p.requires_grad = False

    prior_pi = uniform_distribution(v).to(device)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True).to(device)
    prior_pi.requires_grad = False
    
    emp_tnd_views = torch.tensor(emp_tnd_views).to(device)
    emp_dis_views = torch.tensor(emp_dis_views).to(device)
    
    lamb1, lamb2 = None, None
    if optimise_lambdas:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor, 0.0001, 1.9999)
        lamb1 = torch.nn.Parameter(lamb_tensor)
        
        lamb_tensor2 = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor2, 0.0001, 1.9999)
        lamb2 = torch.nn.Parameter(lamb_tensor2)
        
        all_parameters = list(posterior_Qv) + [posterior_rho, lamb1, lamb2]
    else:
        all_parameters = list(posterior_Qv) + [posterior_rho] 
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')

    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_mv_loss(emp_tnd_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, nt, nd, delta, lamb1, lamb2)
    
        loss.backward() # Backpropagation

        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
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



def compute_loss(emp_tnd, emp_dis, posterior_Q, prior_P, nt, nd, delta, lamb1=None, lamb2=None):
    """
     Compute the loss function for the Majority Vote Learning algorithm.

     Args:
        - emp_tnd (float): The empirical tandem risk for a view.
        - emp_dis (float): Tthe empirical disagreement for a view.
        - posterior_Q (torch.nn.Parameter): The posterior distribution for a view.
        - prior_P (torch.nn.Parameter): The prior distributions for a view.
        - nt (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.

    Returns:
        - tensor: The computed loss value.

     """
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Q = F.softmax(posterior_Q, dim=0)
    
    # Compute the empirical risk
    emp_tandem = torch.sum(torch.sum(emp_tnd * softmax_posterior_Q, dim=0) * softmax_posterior_Q)
    
    # Compute the empirical disagreement
    emp_dis = torch.sum(torch.sum(emp_dis * softmax_posterior_Q, dim=0) * softmax_posterior_Q)
    # print(f"{emp_tandem.item()=}, {emp_dis.item()=}")
    # Compute the Kullback-Leibler divergence
    KL_QP = kl(softmax_posterior_Q, prior_P)
    
    if lamb1 is None or lamb2 is None:
        lamb1 = 2.0 / (torch.sqrt((1.0 * nt * emp_tandem) / (KL_QP + torch.log(2.0 * torch.sqrt(nt) / delta)) + 1.0) + 1.0)
        lamb2 = 2.0 / (torch.sqrt((1.0 * nd * emp_dis) / (KL_QP + torch.log(2.0 * torch.sqrt(nd) / delta)) + 1.0) + 1.0)
    
    loss_term1 = emp_tandem / (1.0 - lamb1 / 2.0) + (2*KL_QP + torch.log((2.0 * torch.sqrt(nt)) / delta)) / (lamb1 * (1.0 - lamb1 / 2.0) * nt)
    loss_term2 = emp_dis / (1.0 - lamb2 / 2.0) + (2*KL_QP + torch.log((2.0 * torch.sqrt(nd)) / delta)) / (lamb2 * (1.0 - lamb2 / 2.0) * nd)

    loss = 2.0*loss_term1 + loss_term2
    
    return loss


def optimizeTND_DIS_torch(emp_tnd, emp_dis, nt, nd, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambdas=False):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_tnd (float): The empirical tandem risk for a view.
        - emp_dis (float): Tthe empirical disagreement for a view.
        - nt (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.

    Returns:
        - tuple: A tuple containing the optimized posterior distribution for a view (posterior_Q).
    """
    
    m = len(emp_tnd)
    
    # Initialisation with the uniform distribution
    prior_P = uniform_distribution(m).to(device)
    posterior_Q = torch.nn.Parameter(prior_P.clone(), requires_grad=True).to(device)
    # We don't need to compute the gradients for the  hyper-prior too
    prior_P.requires_grad = False
    
    # Convert the empirical risks and disagreements to tensors
    emp_tnd = torch.tensor(emp_tnd).to(device)
    emp_dis = torch.tensor(emp_dis).to(device)
    
    lamb1, lamb2 = None, None
    if optimise_lambdas:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor, 0.0001, 1.9999)
        lamb1 = torch.nn.Parameter(lamb_tensor)
        
        lamb_tensor2 = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor2, 0.0001, 1.9999)
        lamb2 = torch.nn.Parameter(lamb_tensor2)
        
        all_parameters = [posterior_Q, lamb1, lamb2]
    else:
        all_parameters = [posterior_Q]
        
    # Optimizer
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_loss(emp_tnd, emp_dis, posterior_Q, prior_P, nt, nd, delta, lamb1, lamb2)
    
        loss.backward() # Backpropagation
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        if optimise_lambdas:
            # Clamping the values of lambdas
            lamb1.data = lamb1.data.clamp(0.0001, 1.9999)
            lamb2.data = lamb2.data.clamp(0.0001, 1.9999)
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
    return softmax_posterior_Q, lamb1, lamb2
