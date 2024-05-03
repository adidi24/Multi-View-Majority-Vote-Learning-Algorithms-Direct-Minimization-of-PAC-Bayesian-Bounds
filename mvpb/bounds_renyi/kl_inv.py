#
# Implements the KL  inv bound in theorem 4.4 using  rd divergence.
#

from math import sqrt, log
import torch
import torch.nn.functional as F

from ..cocob_optim import COCOB

from mvpb.tools import solve_kl_inf, solve_kl_sup
from mvpb.util import uniform_distribution
from mvpb.util import renyi_divergence as rd
from mvpb.util import LogBarrierFunction as lbf
from mvpb.util import kl_inv, klInvFunction

def KLInv_MV(empirical_gibbs_risk, m, RD_QP, RD_rhopi, delta=0.05):
    """ Multi view KL inv bound using Rényi divergence
    """

    phi_r = (RD_QP + RD_rhopi + log((4.0 * sqrt(m)) / delta)) / m
    b = kl_inv(empirical_gibbs_risk, phi_r, "MAX")

    return 2.0*b


def compute_mv_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, delta, alpha=1.1):
    """
     Compute the loss function for the Multi-View Majority Vote Learning algorithm in theorem 4.4

     Args:
        - emp_risks_views (list): A list of empirical risks for each view.
        - posterior_Qv (list): A list of posterior distributions for each view.
        - posterior_rho (tensor): The hyper-posterior distribution for the weights.
        - prior_Pv (list): A list of prior distributions for each view.
        - prior_pi (tensor): The hyper-prior distribution for the weights.
        - ng (int): The number of samples for the risk.
        - delta (float): The confidence parameter.

    Returns:
        - tensor: The computed loss value.

     """
     
    # Apply softmax to ensure that the weights are probability distributions
    softmax_posterior_Qv = [F.softmax(q, dim=0) for q in posterior_Qv]
    softmax_posterior_rho = F.softmax(posterior_rho, dim=0)

    # Compute the empirical risk
    emp_risks = [torch.sum(view * q) for view, q in zip(emp_risks_views, softmax_posterior_Qv)]
    emp_mv_risk = torch.sum(torch.stack(emp_risks) * softmax_posterior_rho)
    
    print(f"{emp_mv_risk.item()=}")

    # Compute the Rényi divergences
    RD_QP = torch.sum(torch.stack([rd(q, p, alpha)  for q, p in zip(softmax_posterior_Qv, prior_Pv)]) * softmax_posterior_rho)
    RD_rhopi = rd(softmax_posterior_rho, prior_pi, alpha)
    
    klinv = klInvFunction.apply
    phi_r = (RD_QP + RD_rhopi + torch.log((4.0 * torch.sqrt(ng)) / delta)) / ng
    loss = klinv(emp_mv_risk, phi_r, "MAX")

    return 2.0*loss, loss


def optimizeKLinv_mv_torch(emp_risks_views, ng, device, max_iter=1000, delta=0.05, eps=10**-9, alpha=1.1, t=1):
    """
    Optimize the value of `lambda` using Pytorch for Multi-View Majority Vote Learning Algorithms.

    Args:
        - emp_risks_views (list): A list of empirical risks for each view.
        - ng (int): The number of samples for the risk.
        - delta (float, optional): The confidence level. Default is 0.05.
        - eps (float, optional): A small value for convergence criteria. Defaults to 10**-9.
        - alpha (float, optional): The Rényi divergence order. Default is 1.1.

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
    
    all_parameters = list(posterior_Qv) + [posterior_rho]
        
    # Optimizer
    optimizer = COCOB(all_parameters)

    prev_loss = float('inf')
    # Optimisation loop
    for i in range(max_iter):
        optimizer.zero_grad()
    
        # Calculating the loss
        loss, constraint = compute_mv_loss(emp_risks_views, posterior_Qv, posterior_rho, prior_Pv, prior_pi, ng, delta, alpha)
        loss += log_barrier(constraint-0.5)
        loss.backward() # Backpropagation
        # print(f"{gamma=}")
    
        # torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
        optimizer.step() # Update the parameters

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
    return softmax_posterior_Qv, softmax_posterior_rho