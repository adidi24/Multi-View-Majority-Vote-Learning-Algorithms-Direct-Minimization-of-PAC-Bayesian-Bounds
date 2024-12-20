#
# Implements the KL  inv bound in theorem 4.4 using  rd divergence.
#

from math import sqrt, log
import torch
import torch.nn.functional as F

from ..cocob_optim import COCOB

from mvpb.util import uniform_distribution
from ..tools import (renyi_divergence as rd,
                        kl,
                        kl_inv,
                        klInvFunction,
                        LogBarrierFunction as lbf)

def KLInv_MV(empirical_gibbs_risk, m, DIV_QP, DIV_rhopi, delta=0.05):
    """ Multi view inverted KL bound using Rényi divergence  (theorem 4.4)
    
    Args:
        - empirical_gibbs_risk (float): The empirical Gibbs risk.
        - m (int): The number of samples.
        - DIV_QP (float): The divergence between the posterior and the prior (default is Rényi divergence).
        - DIV_rhopi (float): The divergence between the hyper-posterior and the hyper-prior (default is Rényi divergence).
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """

    phi_r = (DIV_QP + DIV_rhopi + log((2.0 * sqrt(m)) / delta)) / m
    b = kl_inv(empirical_gibbs_risk, phi_r, "MAX")

    return 2.0*b

def KLInv(empirical_gibbs_risk, m, DIV_QP, delta=0.05):
    """ Majority vote inverted KL bound using Rényi divergence
    
    Args:
        - empirical_gibbs_risk (float): The empirical Gibbs risk.
        - m (int): The number of samples.
        - DIV_QP (float): The divergence between the posterior and the prior (default is KL divergence).
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """

    phi_r = (DIV_QP + log((2.0 * sqrt(m)) / delta)) / m
    b = kl_inv(empirical_gibbs_risk, phi_r, "MAX")

    return 2.0*b


def compute_mv_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, delta, alpha=1.1, alpha_v=None):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 4.4

     Args:
        - emp_risks_views (tensor): A (n_views, n_estimators) tensor of empirical Gibbs risks for each view.
        - posterior_Qv (list): A list of posterior distributions for each view.
        - posterior_rho (tensor): The hyper-posterior distribution for the weights.
        - prior_Pv (list): A list of prior distributions for each view.
        - prior_pi (tensor): The hyper-prior distribution for the weights.
        - ng (int): The number of samples for the risk.
        - delta (float): The confidence parameter.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1. (optimizable if alpha_v is not None)
        - alpha_v (list, optional): A list of optimizable Rényi divergence orders for each view. Default is None.

    Returns:
        - tensor: The computed loss value.

     """
     
    # Apply softmax to ensure that the weights are probability distributions
    log_softmax_posterior_Qv = [F.log_softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_Qv = [torch.exp(q) for q in log_softmax_posterior_Qv]
    log_softmax_posterior_rho = F.log_softmax(posterior_rho, dim=0)
    softmax_posterior_rho = torch.exp(log_softmax_posterior_rho)

    # Compute the empirical risk
    emp_risks = [torch.sum(view * q) for view, q in zip(emp_risks_views, softmax_posterior_Qv)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks) * softmax_posterior_rho)
    
    # print(f"{emp_mv_risk.item()=}")
    if alpha_v is not None:
        # Compute the Rényi divergences with view-specific alpha
        DIV_QP = torch.sum(torch.stack([rd(q, p, a)  for q, p, a in zip(softmax_posterior_Qv, prior_Pv, alpha_v)]) * softmax_posterior_rho)
        DIV_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    else:
        if alpha != 1:
            # Compute the Rényi divergences
            DIV_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
            DIV_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
        else:
            # Compute the KL divergences
            DIV_QP = torch.sum(torch.stack([kl(q, p)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
            DIV_rhopi = kl(softmax_posterior_rho, prior_pi)
    
    klinv = klInvFunction.apply
    phi_r = (DIV_QP + DIV_rhopi + torch.log((2.0 * torch.sqrt(ng)) / delta)) / ng
    loss = klinv(emp_mv_risk, phi_r, "MAX")

    return 2.0*loss, loss


def optimizeKLinv_mv_torch(emp_risks_views, ng, device, max_iter=1000, delta=0.05, eps=10**-9, optimize_alpha=False, alpha=1.1, t=100):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks_views (tensor): A (n_views, n_estimators) tensor of empirical Gibbs risks for each view.
        - ng (int): The number of samples for the risk.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - optimize_alpha (bool, optional): Whether to optimize the alpha parameter. Default is False.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1 (won't be used if optimize_alpha is True).
        - t (int, optional): The Rényi divergence order. Default is 100.

    Returns:
        - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    
    m = len(emp_risks_views[0])
    v = len(emp_risks_views)
    log_barrier = lbf(t)
    
    # Initialisation with the uniform distribution
    prior_Pv = [uniform_distribution(m).to(device)]*v
    posterior_Qv = torch.nn.ParameterList([torch.nn.Parameter(prior_Pv[k].clone(), requires_grad=True).to(device) for k in range(v)])
    for p in prior_Pv:
        # We don't need to compute the gradients for the prior
        p.requires_grad = False

    prior_pi = uniform_distribution(v).to(device)
    posterior_rho = torch.nn.Parameter(prior_pi.clone(), requires_grad=True).to(device)
    # We don't need to compute the gradients for the  hyper-prior too
    prior_pi.requires_grad = False
    
    # Convert the empirical risks to tensor
    emp_risks_views = torch.from_numpy(emp_risks_views).to(device)
    
    alpha_v = None
    if optimize_alpha:
        # Initialisation of beta with zeros tensor with the size as the number of views, hence alpha starts at 1 + exp(0) = 2
        beta_v_tensor = torch.zeros_like(prior_pi).to(device).requires_grad_()
        beta_v = torch.nn.Parameter(beta_v_tensor)
        
        # For the hyper distributions
        beta_tensor = torch.zeros(1).to(device).requires_grad_()
        beta = torch.nn.Parameter(beta_tensor)
    
        all_parameters = list(posterior_Qv) + [posterior_rho, beta_v, beta]
    else:
        all_parameters = list(posterior_Qv) + [posterior_rho]

    # Optimizer
    # optimizer = COCOB(all_parameters)
    optimizer = torch.optim.AdamW(all_parameters, lr=0.1, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,150,250], gamma=0.02)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
        
        if optimize_alpha:
            alpha_v = 1 + torch.exp(beta_v)
            alpha = 1 + torch.exp(beta)
    
        # Calculating the loss
        loss, constraint = compute_mv_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, delta, alpha, alpha_v)
        loss += log_barrier(constraint-0.5)
        loss.backward() # Backpropagation
        # print(f"{gamma=}")
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        scheduler.step()

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
    return softmax_posterior_Qv, softmax_posterior_rho, alpha_v, alpha


def compute_loss(emp_risks, posterior_Q, prior_P, n, delta, alpha=1):
    """
     Compute the loss function for the Majority Vote Learning algorithm

     Args:
        - emp_risks (tensor): A (n_views, n_estimators) tensor of empirical risks.
        - posterior_Q (tensor): The posterior distribution for the estimators.
        - prior_P (tensor): The prior distributions for the estimators.
        - n (int): The number of samples for the risk.
        - delta (float): The confidence parameter.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).


     Returns:
        - tuple: A tuple containing the computed loss value, the joint error constraint and the disagreement constraint.

     """
    
    # Apply softmax to ensure that the weights are probability distributions
    log_softmax_posterior_Q = F.log_softmax(posterior_Q, dim=0)
    softmax_posterior_Q = torch.exp(log_softmax_posterior_Q)
    
    # Compute the empirical risk
    emp_risk = torch.sum(emp_risks * softmax_posterior_Q)

    if alpha != 1:
        # Compute the Rényi divergence
        DIV_QP = rd(softmax_posterior_Q, prior_P, alpha)
    else:
        # Compute the KL divergence
        DIV_QP = kl(softmax_posterior_Q, prior_P)
    
    klinv = klInvFunction.apply
    phi_r = (DIV_QP + torch.log((2.0 * torch.sqrt(n)) / delta)) / n
    loss = klinv(emp_risk, phi_r, "MAX")

    return 2.0*loss, loss


def optimizeKLinv_torch(emp_risks, n, device, max_iter=1000, delta=0.05, eps=10**-9, alpha=1, t=100):
    """
    Optimization using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks (tensor): A (n_views, n_estimators) tensor of empirical risks.
        - n (int): The number of samples for the risk.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).

    Returns:
        - tensor: The optimized posterior distribution (posterior_Q).
    """
    
    m = len(emp_risks)
    
    log_barrier = lbf(t)
    
    # Initialisation with the uniform distribution
    prior_P = uniform_distribution(m).to(device)
    posterior_Q = torch.nn.Parameter(prior_P.clone(), requires_grad=True).to(device)
    # We don't need to compute the gradients for the  hyper-prior too
    prior_P.requires_grad = False
    
    # Convert the empirical risks and disagreements to tensors
    emp_risks = torch.tensor(emp_risks).to(device)
    
    all_parameters = [posterior_Q]
        
    # Optimizer
    # optimizer = COCOB(all_parameters)
    optimizer = torch.optim.AdamW(all_parameters, lr=0.1, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,150,250], gamma=0.02)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint = compute_loss(emp_risks, posterior_Q, prior_P, n, delta, alpha)
        loss += log_barrier(constraint-0.5)
        loss.backward() # Backpropagation
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        scheduler.step()
        # Verify the convergence criteria of the loss
        if torch.abs(prev_loss - loss).item() <= eps:
            print(f"\t Convergence reached after {i} iterations")
            break
    
        prev_loss = loss.item()  # Update the previous loss with the current loss
        # Optional: Display the loss for monitoring
        # print(f"Iteration: {i},\t Loss: {loss.item()}")

    # After the optimization
    with torch.no_grad():
        softmax_posterior_Q = torch.softmax(posterior_Q, dim=0)
    return softmax_posterior_Q