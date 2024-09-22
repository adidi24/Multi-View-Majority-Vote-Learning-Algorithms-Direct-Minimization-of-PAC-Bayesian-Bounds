#
# Implements the DIS  bound with inverted KL in theorem 5.4 using  rd divergence.
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

def DIS_Inv_MV(gibbs_risk, disagreement, ng, nd, DIV_QP, DIV_rhopi, delta=0.05):
    """ Multi-view bound with inverted KL using Rényi divergence  (theorem 5.4)
    
    Args:
        - gibbs_risk(float): The empirical gibbs risk.
        - disagreement (float): The empirical disagreement.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - DIV_QP (float): By default, the Rényi divergence between the posterior and the prior.
        - DIV_rhopi (float): By default, the Rényi divergence between the hyper-posterior and the hyper-prior.
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """
    phi_r = (DIV_QP + DIV_rhopi + log((4.0 * sqrt(ng)) / delta)) / ng
    phi_d = (2.0*(DIV_QP + DIV_rhopi) + log((4.0 * sqrt(nd)) / delta)) / nd
    r = kl_inv(gibbs_risk, phi_r, "MAX")
    d = kl_inv(disagreement, phi_d, "MIN")
    
    b = 4.0*r - 2.0*d
    return b

def DIS_Inv(gibbs_risk, disagreement, ng, nd, DIV_QP, delta=0.05):
    """ Majority vote bound with inverted KL using Rényi divergence  (theorem 5.4)
    
    Args:
        - gibbs_risk(float): The empirical gibbs risk.
        - disagreement (float): The empirical disagreement.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - DIV_QP (float): By default, the KL divergence between the posterior and the prior.
        - delta (float): The confidence parameter.
    
    Returns:
        - float: The computed bound.
    """
    phi_r = (DIV_QP + log((4.0 * sqrt(ng)) / delta)) / ng
    phi_d = (2.0*DIV_QP + log((4.0 * sqrt(nd)) / delta)) / nd
    r = kl_inv(gibbs_risk, phi_r, "MAX")
    d = kl_inv(disagreement, phi_d, "MIN")
    
    b = 4.0*r - 2.0*d
    return b


def compute_mv_loss(grisks_views, dS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, alpha=1.1, alpha_v=None):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 5.4

     Args:
        - grisks_views (tensor): A (n_views, n_estimators) tensor of empirical Gibbs risks for each view.
        - dS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical disagreements for each view.
        - posterior_Qv (list): A list of posterior distributions for each view.
        - posterior_rho (tensor): The hyper-posterior distribution for the views.
        - prior_Pv (list): A list of prior distributions for each view.
        - prior_pi (tensor): The hyper-prior distribution for the weights.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1. (optimizable if alpha_v is not None)
        - alpha_v (list, optional): A list of optimizable Rényi divergence orders for each view. Default is None.


     Returns:
        - tuple: A tuple containing the computed loss value, the gibbs risk constraint and the disagreement constraint.

     """
    nb_views = len(grisks_views)
     
    # Apply softmax to ensure that the weights are probability distributions
    log_softmax_posterior_Qv = [F.log_softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_Qv = [torch.exp(q) for q in log_softmax_posterior_Qv]
    log_softmax_posterior_rho = F.log_softmax(posterior_rho, dim=0)
    softmax_posterior_rho = torch.exp(log_softmax_posterior_rho)

    # Compute the empirical risk
    emp_risks = [torch.sum(view * q) for view, q in zip(grisks_views, softmax_posterior_Qv)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks) * softmax_posterior_rho)
    
    # Compute the empirical disagreement
    dS_v = torch.zeros((nb_views, nb_views), device=dS_views.device)
    for i in range(nb_views):
        for j in range(nb_views):
            dS_v[i, j] = torch.sum(torch.sum(dS_views[i, j]*softmax_posterior_Qv[i], dim=0) * softmax_posterior_Qv[j], dim=0)
    dS_mv =  torch.sum(torch.sum(dS_v*softmax_posterior_rho, dim=0) * softmax_posterior_rho, dim=0)

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
    phi_r = (DIV_QP + DIV_rhopi + torch.log((4.0 * torch.sqrt(ng)) / delta)) / ng
    phi_d = (2.0*(DIV_QP + DIV_rhopi) + torch.log((4.0 * torch.sqrt(nd)) / delta)) / nd
    
    loss_r = klinv(emp_mv_risk, phi_r, "MAX")
    loss_d = klinv(dS_mv, phi_d, "MIN")

    loss = 4.0*loss_r - 2.0*loss_d

    return loss, loss_r, loss_d


def optimizeDIS_Inv_mv_torch(grisks_views, dS_views, ng, nd, device, max_iter=1000, delta=0.05, eps=10**-9, optimize_alpha=False, alpha=1.1, t=100):
    """
    Optimization using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - grisks_views (tensor): A (n_views, n_estimators) tensor of empirical Gibbs risks for each view.
        - dS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical disagreements for each view.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - optimize_alpha (bool, optional): Whether to optimize the alpha parameter. Default is False.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1 (won't be used if optimize_alpha is True).
        - t (float, optional): The parameter for the log barrier function. Default is 0.0001.

    Returns:
        - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    
    assert len(grisks_views) == len(dS_views)
    m = len(grisks_views[0])
    v = len(grisks_views)
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
    
    # Convert the empirical risks and disagreements to tensors
    grisks_views = torch.from_numpy(grisks_views).to(device)
    dS_views = torch.from_numpy(dS_views).to(device)
    
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,150,250], gamma=0.01)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
        
        if optimize_alpha:
            alpha_v = 1 + torch.exp(beta_v)
            alpha = 1 + torch.exp(beta)
    
        # Calculating the loss
        loss, constraint_risk, constraint_dis = compute_mv_loss(grisks_views, dS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, alpha, alpha_v)
        loss += log_barrier(constraint_risk-0.5) + log_barrier(constraint_dis-0.5)
        loss.backward() # Backpropagation
    
        torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        scheduler.step()

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
    return softmax_posterior_Qv, softmax_posterior_rho, alpha_v, alpha



def compute_loss(emp_risks, emp_dis, posterior_Q, prior_P, ng, nd, delta, alpha=1):
    """
     Compute the loss function for the Majority Vote Learning algorithm.

     Args:
        - emp_risks (tensor): A (n_estimators,) tensor of empirical joint errors.
        - emp_dis (tensor): A (n_estimators, n_estimators) tensor of empirical disagreements.
        - posterior_Q (torch.nn.Parameter): The posterior distribution for a view.
        - prior_P (torch.nn.Parameter): The prior distributions for a view.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float): The confidence parameter.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).

    Returns:
        - tensor: The computed loss value.

     """
    # Apply softmax to ensure that the weights are probability distributions
    log_softmax_posterior_Q = F.log_softmax(posterior_Q, dim=0)
    softmax_posterior_Q = torch.exp(log_softmax_posterior_Q)
    
    # Compute the empirical risk
    emp_risk = torch.sum(emp_risks * softmax_posterior_Q)
    
    # Compute the empirical disagreement
    emp_disagreement = torch.sum(torch.sum(emp_dis * softmax_posterior_Q, dim=0) * softmax_posterior_Q)

    if alpha != 1:
        # Compute the Rényi divergence
        DIV_QP = rd(softmax_posterior_Q, prior_P, alpha)
    else:
        # Compute the KL divergence
        DIV_QP = kl(softmax_posterior_Q, prior_P)
    
    klinv = klInvFunction.apply
    phi_r = (DIV_QP + torch.log((4.0 * torch.sqrt(ng)) / delta)) / ng
    phi_d = (2.0*DIV_QP + torch.log((4.0 * torch.sqrt(nd)) / delta)) / nd
    
    loss_r = klinv(emp_risk, phi_r, "MAX")
    loss_d = klinv(emp_disagreement, phi_d, "MIN")

    loss = 4.0*loss_r - 2.0*loss_d

    return loss, loss_r, loss_d


def optimizeDIS_Inv_torch(emp_risks, emp_dis, ng, nd, device, max_iter=1000, delta=0.05, eps=10**-9, alpha=1, t=100):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks (tensor): A (n_estimators,) tensor of empirical joint errors.
        - emp_dis (tensor): A (n_estimators, n_estimators) tensor of empirical disagreements.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1 (KL divergence).
        - t (float, optional): The parameter for the log barrier function. Default is 0.0001.

    Returns:
        - tuple: The optimized posterior distribution for a view (posterior_Q).
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
    emp_dis = torch.tensor(emp_dis).to(device)
    
    all_parameters = [posterior_Q]
        
    # Optimizer
    # optimizer = COCOB(all_parameters)
    optimizer = torch.optim.AdamW(all_parameters, lr=0.1, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,150,250], gamma=0.01)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint_risk, constraint_disagreement = compute_loss(emp_risks, emp_dis, posterior_Q, prior_P, ng, nd, delta, alpha)
        loss += log_barrier(constraint_risk-0.5) + log_barrier(constraint_disagreement-0.5)
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

    # After the optimization: Apply the softmax to the posterior distribution 
    # to ensure that the weights are probability distributions
    with torch.no_grad():
        softmax_posterior_Q = torch.softmax(posterior_Q, dim=0)
    return softmax_posterior_Q
