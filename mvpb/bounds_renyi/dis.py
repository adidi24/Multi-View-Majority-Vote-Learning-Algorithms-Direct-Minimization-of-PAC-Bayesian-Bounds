#
# Implements the DIS bound in theorem 4.6 using  rd divergence.
#

from math import sqrt, log
import torch
import torch.nn.functional as F

from ..cocob_optim import COCOB

from mvpb.tools import solve_kl_inf, solve_kl_sup
from mvpb.util import uniform_distribution
from mvpb.util import renyi_divergence as rd

def DIS(gibbs_risk, disagreement, ng, nd, RD_QP, delta=0.05):
    g_rhs = ( RD_QP + log(4.0*sqrt(ng)/delta) ) / ng
    g_ub  = min(1.0, solve_kl_sup(gibbs_risk, g_rhs))
    
    d_rhs = ( 2.0*RD_QP + log(4.0*sqrt(nd)/delta) ) / nd
    d_lb  = solve_kl_inf(disagreement, d_rhs)
    return min(1.0, 4*g_ub - 2*d_lb)

# Implementation of DIS
def DIS_MV(gibbs_risk, disagreement, ng, nd, RD_QP, RD_rhopi, delta=0.05):
    g_rhs = ( RD_QP +  RD_rhopi + log(4.0*sqrt(ng)/delta) ) / ng
    g_ub  = min(1.0, solve_kl_sup(gibbs_risk, g_rhs))
    
    d_rhs = ( 2.0*(RD_QP + RD_rhopi) + log(4.0*sqrt(nd)/delta)) / nd
    d_lb  = solve_kl_inf(disagreement, d_rhs)
    return min(1.0, 4*g_ub - 2*d_lb)

def compute_mv_loss(emp_risks_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, lamb=None, gamma=None, alpha=1.1):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 4.6

     Args:
        - emp_risks_views (list): A list of empirical risks for each view.
        - emp_dis_views (list): A list of empirical disagreements for each view.
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

    # Compute the Rényi divergences
    RD_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
    RD_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    
    if lamb is None or gamma is None:
        lamb = 2.0 / (torch.sqrt((2.0 * ng * emp_mv_risk) / (RD_QP + RD_rhopi + torch.log(4.0 * torch.sqrt(ng) / delta)) + 1.0) + 1.0)
        gamma = torch.sqrt(2.0 * (RD_QP + RD_rhopi + torch.log((4.0*torch.sqrt(nd))/delta)) / (emp_mv_dis*nd))
    
    phi = emp_mv_risk / (1.0 - lamb / 2.0) + (RD_QP + RD_rhopi + torch.log((4.0 * torch.sqrt(ng)) / delta)) / (lamb * (1.0 - lamb / 2.0) * ng)
    dis_term = (1-gamma/2.0) * emp_mv_dis - 2*(RD_QP + RD_rhopi) + torch.log((4.0 * torch.sqrt(nd)) / delta) / (gamma*nd)

    loss = 4.0*phi - 2.0*dis_term
    
    return loss


def optimizeDIS_mv_torch(emp_risks_views, emp_dis_views, ng, nd, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambda_gamma=False, alpha=1.1):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks_views (list): A list of empirical risks for each view.
        - emp_dis_views (list): A list of empirical disagreements for each view.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.

    Returns:
        - tuple: A tuple containing the optimized posterior distributions for each view (posterior_Qv) and the optimized hyper-posterior distribution (posterior_rho).
    """
    
    assert len(emp_risks_views) == len(emp_dis_views)
    m = len(emp_risks_views[0])
    v = len(emp_risks_views)
    
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
    emp_risks_views = torch.tensor(emp_risks_views).to(device)
    emp_dis_views = torch.tensor(emp_dis_views).to(device)
    
    lamb, gamma = None, None
    if optimise_lambda_gamma:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor, 0.0001, 1.9999)
        lamb = torch.nn.Parameter(lamb_tensor)
        
        gamma_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(gamma_tensor, 0.0001)
        gamma = torch.nn.Parameter(gamma_tensor)
        
        all_parameters = list(posterior_Qv) + [posterior_rho, lamb, gamma]
    else:
        all_parameters = list(posterior_Qv) + [posterior_rho]
        
    # Optimizer
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_mv_loss(emp_risks_views, emp_dis_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, nd, delta, lamb, gamma, alpha)
    
        loss.backward() # Backpropagation
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        if optimise_lambda_gamma:
            # Clamping the values of lambda and gamma
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



def compute_loss(emp_risks, emp_dis, posterior_Q, prior_P, ng, nd, delta, lamb=None, gamma=None, alpha=1.1):
    """
     Compute the loss function for the Majority Vote Learning algorithm.

     Args:
        - emp_risks (float): The empirical risk for a view.
        - emp_dis (float): Tthe empirical disagreement for a view.
        - posterior_Q (torch.nn.Parameter): The posterior distribution for a view.
        - prior_P (torch.nn.Parameter): The prior distributions for a view.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
        - delta (float): The confidence parameter.
        - lamb (float): lambda.

    Returns:
        - tensor: The computed loss value.

     """
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Q = F.softmax(posterior_Q, dim=0)
    
    # Compute the empirical risk
    emp_risk = torch.sum(emp_risks * softmax_posterior_Q)
    
    # Compute the empirical disagreement
    emp_disagreement = torch.sum(torch.sum(emp_dis * softmax_posterior_Q) * softmax_posterior_Q)

    # Compute the Rényi divergence
    RD_QP = rd(softmax_posterior_Q, prior_P, alpha)
    
    if lamb is None or gamma is None:
        lamb = 2.0 / (torch.sqrt((2.0 * ng * emp_risk) / (RD_QP + torch.log(4.0 * torch.sqrt(ng) / delta)) + 1.0) + 1.0)
        gamma = torch.sqrt(2.0 * (RD_QP + torch.log((4.0*torch.sqrt(nd))/delta)) / (emp_disagreement*nd))
    
    phi = emp_risk / (1.0 - lamb / 2.0) + RD_QP + torch.log((4.0 * torch.sqrt(ng)) / delta) / (lamb * (1.0 - lamb / 2.0) * ng)
    dis_term = (1-gamma/2.0) * emp_disagreement - (2*RD_QP + torch.log((4.0 * torch.sqrt(nd)) / delta)) / (gamma*nd)

    loss = 4.0*phi - 2.0*dis_term
    
    return loss


def optimizeDIS_torch(emp_risks, emp_dis, ng, nd, device, max_iter=1000, delta=0.05, eps=10**-9, optimise_lambda_gamma=False, alpha=1.1):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks (float): The empirical risk for a view.
        - emp_dis (float): Tthe empirical disagreement for a view.
        - ng (int): The number of samples for the risk.
        - nd (int): The number of samples for the disagreement.
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
    emp_dis = torch.tensor(emp_dis).to(device)
    
    lamb, gamma = None, None
    if optimise_lambda_gamma:
        # Initialisation of lambda with a random value between 0 and 2 (exclusive)
        lamb_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(lamb_tensor, 0.0001, 1.9999)
        lamb = torch.nn.Parameter(lamb_tensor)
        
        gamma_tensor = torch.empty(1).to(device).requires_grad_()
        # Apply the uniform distribution
        torch.nn.init.uniform_(gamma_tensor, 0.0001)
        gamma = torch.nn.Parameter(gamma_tensor)
        
        all_parameters = [posterior_Q, lamb, gamma]
    else:
        all_parameters = [posterior_Q]
        
    # Optimizer
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss = compute_loss(emp_risks, emp_dis, posterior_Q, prior_P, ng, nd, delta, lamb, gamma, alpha)
    
        loss.backward() # Backpropagation
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters
        if optimise_lambda_gamma:
            # Clamping the values of lambda and gamma
            lamb.data = lamb.data.clamp(0.0001, 1.9999)
            gamma.data = gamma.data.clamp_min(0.0001)
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
    return softmax_posterior_Q, lamb, gamma
