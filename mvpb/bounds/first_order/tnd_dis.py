#
# Implements the TND and DIS bound in theorem 4.5
#

from math import log, sqrt

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..cocob_optim import COCOB


from mvpb.util import uniform_distribution
from ..tools import (renyi_divergence as rd,
                        kl,
                        LogBarrierFunction as lbf,
                        solve_kl_sup)

# Implementation of DIS
def TND_DIS_MV(eS, dS, ne, nd, DIV_QP, DIV_rhopi, delta=0.05):
    """ Multi-view bound with inverted KL using Rényi divergence  (theorem 4.3)
    
    Args:
        - eS(float): The empirical joint error.
        - dS (float): The empirical disagreement.
        - ne (int): The number of samples for the joint error.
        - nd (int): The number of samples for the disagreement.
        - DIV_QP (float): By default, the Rényi divergence between the posterior and the prior.
        - DIV_rhopi (float): By default, the Rényi divergence between the hyper-posterior and the hyper-prior.
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """
    t_rhs = ( 2.0*(DIV_QP +  DIV_rhopi)+ log(4.0*sqrt(ne)/delta) ) / ne
    t_ub  = min(1.0, solve_kl_sup(eS, t_rhs))
    
    d_rhs = ( 2.0*(DIV_QP + DIV_rhopi) + log(4.0*sqrt(nd)/delta) ) / nd
    d_ub  = solve_kl_sup(dS, d_rhs)
    return min(1.0, 2*t_ub +  d_ub)

def TND_DIS(eS, dS, ne, nd, DIV_QP, delta=0.05):
    """ Majority vote bound using Rényi divergence
    
    Args:
        - eS(float): The empirical joint error.
        - dS (float): The empirical disagreement.
        - ne (int): The number of samples for the joint error.
        - nd (int): The number of samples for the disagreement.
        - DIV_QP (float): By default, the KL divergence between the posterior and the prior.
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """
    t_rhs = ( 2.0*DIV_QP + log(4.0*sqrt(ne)/delta) ) / ne
    t_ub  = min(1.0, solve_kl_sup(eS, t_rhs))
    
    d_rhs = ( 2.0*DIV_QP + log(4.0*sqrt(nd)/delta) ) / nd
    d_lb  = solve_kl_sup(dS, d_rhs)
    return min(1.0, 2*t_ub + d_lb)


def compute_mv_loss(eS_views, dS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, nt, nd, delta, lamb1=None, lamb2=None, alpha=1.1):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 4.3

     Args:
        - eS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical joint errors for each view.
        - dS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical disagreements for each view.
        - posterior_Qv (list): A list of posterior distributions for each view.
        - posterior_rho (tensor): The hyper-posterior distribution for the weights.
        - prior_Pv (list): A list of prior distributions for each view.
        - prior_pi (tensor): The hyper-prior distribution for the weights.
        - ne (int): The number of samples for the joint error.
        - nd (int): The number of samples for the disagreement.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.


     Returns:
         - tuple: A tuple containing the computed loss value, the joint error constraint and the disagreement constraint.

     """
    nb_views = len(eS_views)
    
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical joint error
    eS_v = torch.zeros((nb_views, nb_views), device=eS_views.device)
    for i in range(nb_views):
        for j in range(nb_views):
            eS_v[i, j] = torch.sum(torch.sum(eS_views[i, j]*softmax_posterior_Qv[i], dim=0) * softmax_posterior_Qv[j], dim=0)
    eS_mv =  torch.sum(torch.sum(eS_v*softmax_posterior_rho, dim=0) * softmax_posterior_rho, dim=0)
    
    # Compute the empirical disagreement
    dis_v = torch.zeros((nb_views, nb_views), device=dS_views.device)
    for i in range(nb_views):
        for j in range(nb_views):
            dis_v[i, j] = torch.sum(torch.sum(dS_views[i, j]*softmax_posterior_Qv[i], dim=0) * softmax_posterior_Qv[j], dim=0)
    dS_mv =  torch.sum(torch.sum(dis_v*softmax_posterior_rho, dim=0) * softmax_posterior_rho, dim=0)

    if alpha != 1:
        # Compute the Rényi divergences
        DIV_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
        DIV_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    else:
        # Compute the KL divergences
        DIV_QP = torch.sum(torch.stack([kl(q, p)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
        DIV_rhopi = kl(softmax_posterior_rho, prior_pi)
    
    if lamb1 is None or lamb2 is None:
        lamb1 = 2.0 / (torch.sqrt((1.0 * nt * eS_mv) / (DIV_QP + DIV_rhopi + torch.log(2.0 * torch.sqrt(nt) / delta)) + 1.0) + 1.0)
        lamb2 = 2.0 / (torch.sqrt((1.0 * nd * dS_mv) / (DIV_QP + DIV_rhopi + torch.log(2.0 * torch.sqrt(nd) / delta)) + 1.0) + 1.0)
    
    loss_term_e = eS_mv / (1.0 - lamb1 / 2.0) + (2*(DIV_QP + DIV_rhopi) + torch.log((4.0 * torch.sqrt(nt)) / delta)) / (lamb1 * (1.0 - lamb1 / 2.0) * nt)
    loss_term_d = dS_mv / (1.0 - lamb2 / 2.0) + (2*(DIV_QP + DIV_rhopi) + torch.log((4.0 * torch.sqrt(nd)) / delta)) / (lamb2 * (1.0 - lamb2 / 2.0) * nd)

    loss = 2.0*loss_term_e + loss_term_d
    
    return loss, loss_term_e, loss_term_d


def optimizeTND_DIS_mv_torch(eS_views, dS_views, ne, nd, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambdas=False, alpha=1.1,t=1):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - eS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical joint errors for each view.
        - dS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical disagreements for each view.
        - ne (int): The number of samples for the joint error.
        - nd (int): The number of samples for the disagreement.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.
        - t (float, optional): Controls the steepness and sensitivity of the barrier. Higher values make the barrier more aggressive. Default is 100.


    Returns:
        - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    log_barrier = lbf(t)
    assert len(eS_views) == len(dS_views)
    m = len(eS_views[0, 0])
    v = len(eS_views)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m).to(device)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True).to(device) for k in range(v)])
    for p in prior_Pv:
        p.requires_grad = False

    prior_pi = uniform_distribution(v).to(device)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True).to(device)
    prior_pi.requires_grad = False
    
    eS_views = torch.from_numpy(eS_views).to(device)
    dS_views = torch.from_numpy(dS_views).to(device)
    
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
    
    # optimizer = COCOB(all_parameters)
    optimizer = torch.optim.AdamW(all_parameters, lr=0.1, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,100], gamma=0.01)

    prev_loss = float('inf')

    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint_joint_error, constraint_disagreement = compute_mv_loss(eS_views, dS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ne, nd, delta, lamb1, lamb2, alpha)
        loss += log_barrier(constraint_joint_error-0.25) + log_barrier(constraint_disagreement-(2*(torch.sqrt(constraint_joint_error)-constraint_joint_error)))
        loss.backward() # Backpropagation


        # torch.nn.utils.clip_grad_norm_(all_parameters, 5.0)
        optimizer.step() # Update the parameters
        scheduler.step()
        
        if optimise_lambdas:
            lamb1.data = lamb1.data.clamp(0.0001, 1.9999)
            lamb2.data = lamb2.data.clamp(0.0001, 1.9999)

        # Verify the convergence criteria of the loss
        if torch.abs(prev_loss - loss).item() <= eps:
            print(f"\t Convergence reached after {i} iterations")
            break
        
        prev_loss = loss.item()  # Update the previous loss with the current lossi
        # Optional: Display the loss for monitoring
        # print(f"Iteration: {i},\t Loss: {loss.item()}")

    # After the optimization
    with torch.no_grad():
        softmax_posterior_Qv = [torch.softmax(q, dim=0) for q in posterior_Qv]
        softmax_posterior_rho = torch.softmax(posterior_rho, dim=0)
    return softmax_posterior_Qv, softmax_posterior_rho, lamb1, lamb2



def compute_loss(eS, dS, posterior_Q, prior_P, ne, nd, delta, lamb1=None, lamb2=None, alpha=1, t=100):
    """
     Compute the loss function for the Majority Vote Learning algorithm.

     Args:
        - eS (tensor): A (n_estimators, n_estimators) tensor of empirical joint errors.
        - dS (tensor): A (n_estimators, n_estimators) tensor of empirical disagreements.
        - posterior_Q (torch.nn.Parameter): The posterior distribution for a view.
        - prior_P (torch.nn.Parameter): The prior distributions for a view.
        - ne (int): The number of samples for the joint error.
        - nd (int): The number of samples for the disagreement.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).

    Returns:
        - tensor: The computed loss value.

     """
    

    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Q = F.softmax(posterior_Q, dim=0)
    
    # Compute the empirical joint error
    eS = torch.sum(torch.sum(eS * softmax_posterior_Q, dim=0) * softmax_posterior_Q)
    
    # Compute the empirical disagreement
    dS = torch.sum(torch.sum(dS * softmax_posterior_Q, dim=0) * softmax_posterior_Q)
    
    if alpha != 1:
        # Compute the Rényi divergence
        DIV_QP = rd(softmax_posterior_Q, prior_P, alpha)
    else:
        # Compute the KL divergence
        DIV_QP = kl(softmax_posterior_Q, prior_P)
    
    if lamb1 is None or lamb2 is None:
        lamb1 = 2.0 / (torch.sqrt((1.0 * ne * eS) / (DIV_QP + torch.log(2.0 * torch.sqrt(ne) / delta)) + 1.0) + 1.0)
        lamb2 = 2.0 / (torch.sqrt((1.0 * nd * dS) / (DIV_QP + torch.log(2.0 * torch.sqrt(nd) / delta)) + 1.0) + 1.0)
    
    loss_term1 = eS / (1.0 - lamb1 / 2.0) + (2*DIV_QP + torch.log((2.0 * torch.sqrt(ne)) / delta)) / (lamb1 * (1.0 - lamb1 / 2.0) * ne)
    loss_term2 = dS / (1.0 - lamb2 / 2.0) + (2*DIV_QP + torch.log((2.0 * torch.sqrt(nd)) / delta)) / (lamb2 * (1.0 - lamb2 / 2.0) * nd)

    loss = 2.0*loss_term1 + loss_term2
    
    return loss


def optimizeTND_DIS_torch(eS, dS, ne, nd, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambdas=False, alpha=1):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - eS (tensor): A (n_estimators, n_estimators) tensor of empirical joint errors.
        - dS (tensor): A (n_estimators, n_estimators) tensor of empirical disagreements.
        - ne (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).

    Returns:
        - tuple: A tuple containing the optimized posterior distribution for a view (posterior_Q).
    """
    
    m = len(eS)
    
    # Initialisation with the uniform distribution
    prior_P = uniform_distribution(m).to(device)
    posterior_Q = torch.nn.Parameter(prior_P.clone(), requires_grad=True).to(device)
    # We don't need to compute the gradients for the  hyper-prior too
    prior_P.requires_grad = False
    
    # Convert the empirical risks and disagreements to tensors
    eS = torch.tensor(eS).to(device)
    dS = torch.tensor(dS).to(device)
    
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
        loss = compute_loss(eS, dS, posterior_Q, prior_P, ne, nd, delta, lamb1, lamb2, alpha)
    
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
