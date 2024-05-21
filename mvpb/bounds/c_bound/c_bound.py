#
# Implements the C-bound with inverted KL in theorem 6.1 using  rd divergence.
#

from math import sqrt, log, isinf, isnan
import torch
import torch.nn.functional as F

from ..cocob_optim import COCOB

from tqdm import tqdm

from mvpb.util import uniform_distribution
from ..tools import (renyi_divergence as rd,
                        kl,
                        kl_inv,
                        klInvFunction,
                        LogBarrierFunction as lbf)

def Cbound_MV(gibbs_risk, disagreement, ng, nd, DIV_QP, DIV_rhopi, delta=0.05):
    """ Multi-view C-bound with inverted KL using Rényi divergence  (theorem 6.1)
    
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
    r = min(0.5, r)
    
    d = kl_inv(disagreement, phi_d, "MIN")
    d = max(0.0, d)
    
    cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
    if(isnan(cb) or isinf(cb)):
        cb = 1.0
    return cb

def Cbound(gibbs_risk, disagreement, ng, nd, DIV_QP, delta=0.05):
    """ Majority vote bound with inverted KL using Rényi divergence  (theorem 6.1)
    
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
    r = min(0.5, r)
    
    d = kl_inv(disagreement, phi_d, "MIN")
    d = max(0.0, d)
    
    cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
    if(isnan(cb) or isinf(cb)):
        cb = 1.0
    return cb


def compute_mv_loss(grisks_views, dS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, alpha=1.1):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 6.1

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
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.


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
    dS_v = torch.zeros((nb_views, nb_views), device=grisks_views.device)
    for i in range(nb_views):
        for j in range(nb_views):
            dS_v[i, j] = torch.sum(torch.sum(dS_views[i, j]*softmax_posterior_Qv[i], dim=0) * softmax_posterior_Qv[j], dim=0)
    dS_mv =  torch.sum(torch.sum(dS_v*softmax_posterior_rho, dim=0) * softmax_posterior_rho, dim=0)

    # print(f"emp_mv_risk: {emp_mv_risk}, emp_disagreement: {dS_mv}")
    
    if alpha != 1:
        # Compute the Rényi divergences
        DIV_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
        DIV_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    else:
        # Compute the KL divergences
        DIV_QP = torch.sum(torch.stack([kl(q, p)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
        DIV_rhopi = kl(softmax_posterior_rho, prior_pi)
    
    # print(f"DIV_QP: {DIV_QP}, DIV_rhopi: {DIV_rhopi}")
    
    klinv = klInvFunction.apply
    phi_r = (DIV_QP + DIV_rhopi + torch.log((4.0 * torch.sqrt(ng)) / delta)) / ng
    phi_d = (2.0*(DIV_QP + DIV_rhopi) + torch.log((4.0 * torch.sqrt(nd)) / delta)) / nd
    
    # print(f"phi_r: {phi_r}, phi_d: {phi_d}")
    
    loss_r = klinv(emp_mv_risk, phi_r, "MAX")
    loss_d = klinv(dS_mv, phi_d, "MIN")

    loss_r = torch.min(torch.tensor(0.5).to(loss_r.device), loss_r)
    loss_d = torch.max(torch.tensor(0.0).to(loss_d.device), loss_d)
    
    # Compute the C-bound
    loss = (1.0-((1.0-2.0*loss_r)**2.0)/(1.0-2.0*loss_d))
    
    if(torch.isnan(loss) or torch.isinf(loss)):
        loss = torch.tensor(1.0, requires_grad=True)
    
    # print(f"loss: {loss}------------------")
        
    return loss, loss_r, loss_d


def optimizeCBound_mv_torch(grisks_views, dS_views, ng, nd, device, max_iter=1000, delta=0.05, eps=10**-9, alpha=1.1, t=100):
    """
    Optimization using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - grisks_views (tensor): A (n_views, n_estimators) tensor of empirical Gibbs risks for each view.
        - dS_views (tensor): A (n_views, n_views, n_estimators, n_estimators) tensor of empirical disagreements for each view.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.
        - t (float, optional): The parameter for the log barrier function. Default is 0.0001.

    Returns:
        - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """

    print(f"{device=}")
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
    grisks_views = torch.tensor(grisks_views).to(device)
    dS_views = torch.tensor(dS_views).to(device)
    
    all_parameters = list(posterior_Qv) + [posterior_rho]
        
    # Optimizer
    # optimizer = COCOB(all_parameters)
    # optimizer = torch.optim.AdamW(all_parameters, lr=0.01, weight_decay=0.05)
    optimizer = torch.optim.AdamW(all_parameters, lr=0.01, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80,150,250], gamma=0.002)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint_risk, constraint_dis = compute_mv_loss(grisks_views, dS_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, alpha)
        loss += log_barrier(constraint_risk-0.5) + log_barrier(constraint_dis-0.5)
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
        softmax_posterior_Qv = [torch.softmax(q, dim=0) for q in posterior_Qv]
        softmax_posterior_rho = torch.softmax(posterior_rho, dim=0)
    return softmax_posterior_Qv, softmax_posterior_rho



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
    # softmax_posterior_Q = F.softmax(posterior_Q, dim=0)
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

    loss_r = torch.min(torch.tensor(0.5).to(loss_r.device), loss_r)
    loss_d = torch.max(torch.tensor(0.0).to(loss_d.device), loss_d)
    
    # Compute the C-bound
    loss = (1.0-((1.0-2.0*loss_r)**2.0)/(1.0-2.0*loss_d))
    
    if(torch.isnan(loss) or torch.isinf(loss)):
        loss = torch.tensor(1.0, requires_grad=True)
        
    return loss, loss_r, loss_d


def optimizeCBound_torch(emp_risks, emp_dis, ng, nd, device, max_iter=1000, delta=0.05, eps=10**-9, alpha=1, t=100):
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
