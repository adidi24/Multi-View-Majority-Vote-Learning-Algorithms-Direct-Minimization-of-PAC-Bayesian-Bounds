#
# Implements the TND bound with inverted KL in theorem 5.3.
#

from math import log, sqrt

import torch
import torch.nn.functional as F
import torch.optim as optim

from ..cocob_optim import COCOB

from mvpb.util import uniform_distribution
from ..tools import (renyi_divergence as rd,
                        kl,
                        kl_inv,
                        klInvFunction,
                        LogBarrierFunction as lbf)


def TND_Inv_MV(eS_mv, ne, DIV_QP, DIV_rhopi, delta=0.05):
    """ Multi-view bound with inverted KL using Rényi divergence  (theorem 5.3)
    
    Args:
        - eS_mv (float): The empirical joint error.
        - ne (int): The number of samples for the joint error.
        - DIV_QP (float): By default, the Rényi divergence between the posterior and the prior.
        - DIV_rhopi (float): By default, the Rényi divergence between the hyper-posterior and the hyper-prior.
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """

    phi_e = (2.0*(DIV_QP + DIV_rhopi) + log((2.0 * sqrt(ne)) / delta)) / ne
    e = kl_inv(eS_mv, phi_e, "MAX")
    
    b = 4.0*e
    return b

def TND_Inv(eS, ne, DIV_QP, delta=0.05):
    """ Majority vote bound with inverted KL using Rényi divergence
    
    Args:
        - eS (float): The empirical joint error.
        - ne (int): The number of samples for the joint error.
        - DIV_QP (float): By default, the KL divergence between the posterior and the prior.
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """

    phi_e = (2.0*DIV_QP + log((2.0 * sqrt(ne)) / delta)) / ne
    e = kl_inv(eS, phi_e, "MAX")
    
    b = 4.0*e
    return b

def compute_mv_loss(eS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, alpha=1.1):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 5.3

     Args:
        - eS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical joint errors for each view.
        - posterior_Qv (list): A list of posterior distributions for each view.
        - posterior_rho (tensor): The hyper-posterior distribution for the weights.
        - prior_Pv (list): A list of prior distributions for each view.
        - prior_pi (tensor): The hyper-prior distribution for the weights.
        - n (int): The number of samples.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.

    Returns:
        - tensor: The computed loss value.

     """ 
    nb_views = len(eS_views)
    
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical joint error
    eS_v = torch.zeros((nb_views, nb_views))
    for i in range(nb_views):
        for j in range(nb_views):
            eS_v[i, j] = torch.sum(torch.sum(eS_views[i, j]*softmax_posterior_Qv[i], dim=0) * softmax_posterior_Qv[j])
    eS_mv =  torch.sum(torch.sum(eS_v*softmax_posterior_rho, dim=0) * softmax_posterior_rho)

    if alpha != 1:
        # Compute the Rényi divergences
        DIV_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
        DIV_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    else:
        # Compute the KL divergences
        DIV_QP = torch.sum(torch.stack([kl(q, p)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
        DIV_rhopi = kl(softmax_posterior_rho, prior_pi)
    
    klinv = klInvFunction.apply
    phi_e = (2.0*(DIV_QP + DIV_rhopi) + torch.log((2.0 * torch.sqrt(n)) / delta)) / n
    
    loss_e = klinv(eS_mv, phi_e, "MAX")

    loss = 4.0*loss_e

    return loss, loss_e


def optimizeTND_Inv_mv_torch(eS_views, n, device, max_iter=1000, delta=0.05, eps=10**-9, alpha=1.1, t=1.0):
    """
    Optimization using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - eS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical joint errors for each view.
        - n (list): The number of samples.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.

    Returns:
        - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    
    m = len(eS_views[0, 0])
    v = len(eS_views)
    
    log_barrier = lbf(t)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m).to(device)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True).to(device) for k in range(v)])
    for p in prior_Pv:
        p.requires_grad = False

    prior_pi = uniform_distribution(v).to(device)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True).to(device)
    prior_pi.requires_grad = False
    
    eS_views = torch.from_numpy(eS_views).to(device)
    
    all_parameters = list(posterior_Qv) + [posterior_rho] 
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')

    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint_joint_error = compute_mv_loss(eS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, n, delta, alpha)
        loss += log_barrier(constraint_joint_error-0.25)
        loss.backward() # Backpropagation


        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters

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
    return softmax_posterior_Qv, softmax_posterior_rho



def compute_loss(eS, posterior_Q, prior_P, n, delta, alpha=1):
    """
     Compute the loss function for the Majority Vote Learning algorithm.

     Args:
        - eS (tensor): A (n_estimators, n_estimators) tensor of empirical joint errors for a view.
        - posterior_Q (torch.nn.Parameter): The posterior distribution for a view.
        - prior_P (torch.nn.Parameter): The prior distributions for a view.
        - n (int): The number of samples for the risk.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).

    Returns:
        - tensor: The computed loss value.

     """
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Q = F.softmax(posterior_Q, dim=0)
    
    # Compute the empirical risk
    eS = torch.sum(torch.sum(eS * softmax_posterior_Q, dim=0)*softmax_posterior_Q)

    if alpha != 1:
        # Compute the Rényi divergence
        DIV_QP = rd(softmax_posterior_Q, prior_P, alpha)
    else:
        # Compute the KL divergence
        DIV_QP = kl(softmax_posterior_Q, prior_P)
    
    klinv = klInvFunction.apply
    phi_e = (2.0*DIV_QP + torch.log((2.0 * torch.sqrt(n)) / delta)) / n
    
    loss_e = klinv(eS, phi_e, "MAX")

    loss = 4.0*loss_e

    return loss, loss_e


def optimizeTND_Inv_torch(eS, n, device, max_iter=1000, delta=0.05, eps=10**-9, alpha=1, t=1.0):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - eS_views (tensor): A (n_estimators, n_estimators) tensor of empirical joint errors for a view.
        - n (int): The number of samples for the risk.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).

    Returns:
        - tuple: The optimized posterior distribution for a view (posterior_Q).
    """
    
    m = len(eS)
    
    log_barrier = lbf(t)
    
    # Initialisation with the uniform distribution
    prior_P = uniform_distribution(m).to(device)
    posterior_Q = torch.nn.Parameter(prior_P.clone(), requires_grad=True).to(device)
    # We don't need to compute the gradients for the  hyper-prior too
    prior_P.requires_grad = False
    
    # Convert the empirical risks and disagreements to tensors
    eS = torch.tensor(eS).to(device)
    
    all_parameters = [posterior_Q]
        
    # Optimizer
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint_joint_error = compute_loss(eS, posterior_Q, prior_P, n, delta, alpha)
        loss += log_barrier(constraint_joint_error-0.25)
        loss.backward() # Backpropagation
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters

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
    return softmax_posterior_Q
