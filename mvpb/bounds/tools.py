#-*- coding:utf-8 -*-
""" Tools for bound computation.
Implementation is taken from paper:

Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

The file is renamed from the file pac_bound_tools.py
The original documentation for each function is preserved.

Original documentation:
---
Various functions imported by bound computation files pac_bound_{0,1,1p,2,2p}.py.

See the related paper:
Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

http://graal.ift.ulaval.ca/majorityvote/ 
"""

import math
import numpy as np
from scipy import optimize

import torch.nn.functional as F

def validate_inputs(empirical_gibbs_risk, empirical_disagreement=None, m=None, KLQP=None, delta=0.05):
    """
    This utility function validates if entry parameters are plausible when computing
    PAC-Bayesian bounds.
    """
    is_valid = [True]
    def handle_error(msg):
        print('INVALID INPUT: ' + msg)
        is_valid[0] = False

    if empirical_gibbs_risk < 0.0 or empirical_gibbs_risk >= 0.5:
        handle_error( 'empirical_gibbs_risk must lies in [0.0,0.5)' )
    if empirical_disagreement is not None:
        if empirical_disagreement < 0.0 or empirical_disagreement >= 0.5:
            handle_error( 'empirical_disagreement must lies in [0.0,0.5)' )
        if empirical_disagreement > 2*empirical_gibbs_risk*(1.0-empirical_gibbs_risk):
            handle_error( 'invalid variance, i.e., empirical_disagreement > 2*empirical_gibbs_risk*(1.0-empirical_gibbs_risk)' )
    if m is not None and m <=0:
        handle_error( 'm must be strictly positive.' )
    if KLQP is not None and KLQP < 0.0:
        handle_error( 'KLQP must be positive.' )
    if delta <= 0.0 or delta >= 0.5:
        handle_error( 'delta must lies in (0.0, 1.0)' )

    return is_valid[0]

def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([ q*math.log(q/p) if q > 0. else 0. for q,p in zip(Q,P) ])


def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.-q], [p, 1.-p])


def KL_trinomial(q1, q2, p1, p2):
    """
    Compute the KL-divergence between two multinomial distributions (Q and P)
    with three possible events, where Q=(q1,q2,1-q1-q2), P=(p1,p2,1-p1-p2).
    """
    return KL([q1, q2, 1.-q1-q2], [p1, p2,  1.-p1-p2])

def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0-1e-9) <= 0.0:
        return 1.0-1e-9
    else:
        return optimize.brentq(f, q, 1.0-1e-11)

def solve_kl_inf(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x < q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1e-9) <= 0.0:
        return 1e-9
    else:
        return optimize.brentq(f, 1e-11, q)


def kl(Q, P):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions Q and P.
    
    Args:
        Q (torch.Tensor): The first probability distribution.
        P (torch.Tensor): The second probability distribution.
    
    Returns:
        torch.Tensor: The KL divergence between Q and P.
    """
    assert Q.size() == P.size(), "Distributions must have the same size"
    return F.kl_div(Q.log(), P, reduction='sum')

import torch

def renyi_divergence(Q, P, alpha):
    """
    Compute the Renyi divergence between two probability distributions.

    Args:
        Q (torch.Tensor): The first probability distribution.
        P (torch.Tensor): The second probability distribution.
        alpha (float): The parameter for Renyi divergence.

    Returns:
        torch.Tensor: The Renyi divergence between Q and P.
    """
    assert Q.size() == P.size(), "Distributions must have the same size"

    # Compute the Renyi divergence
    divergence = torch.log(torch.sum(torch.pow(P, alpha) * torch.pow(Q, 1 - alpha)))

    return divergence

###############################################################################
def kl_inv(q, epsilon, mode, tol=1e-9, nb_iter_max=1000):
    """
    Solve the optimization problem min{ p in [0, 1] | kl(q||p) <= epsilon }
    or max{ p in [0,1] | kl(q||p) <= epsilon } for q and epsilon fixed using PyTorch

    Args:
        q (float or torch.Tensor): The parameter q of the kl divergence
        epsilon (float or torch.Tensor): The upper bound on the kl divergence
        tol (float, optional): The precision tolerance of the solution
        nb_iter_max (int, optional): The maximum number of iterations
    """
    assert mode == "MIN" or mode == "MAX"
    q = torch.tensor(q, dtype=torch.float)
    epsilon = torch.tensor(epsilon, dtype=torch.float)
    assert q >= 0 and q <= 1
    assert epsilon > 0.0

    def kl(q, p):
        """
        Compute the KL divergence between two Bernoulli distributions
        (denoted kl divergence) using PyTorch

        Parameters
        ----------
        q: torch.Tensor
            The parameter of the posterior Bernoulli distribution
        p: torch.Tensor
            The parameter of the prior Bernoulli distribution
        """
        return q * torch.log(q / p) + (1 - q) * torch.log((1 - q) / (1 - p))

    # We optimize the problem with the bisection method
    if mode == "MAX":
        p_max = 1.0
        p_min = q
    else:
        p_max = q
        p_min = torch.tensor(10.0**-9, dtype=torch.float)

    for _ in range(nb_iter_max):
        p = (p_min + p_max) / 2.0

        if kl(q, p) == epsilon or (p_max - p_min) / 2.0 < tol:
            return p.item()

        if mode == "MAX" and kl(q, p) > epsilon:
            p_max = p
        elif mode == "MAX" and kl(q, p) < epsilon:
            p_min = p
        elif mode == "MIN" and kl(q, p) > epsilon:
            p_min = p
        elif mode == "MIN" and kl(q, p) < epsilon:
            p_max = p

    return p.item()


###############################################################################


class klInvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, epsilon, mode):
        assert mode == "MIN" or mode == "MAX"
        assert isinstance(q, torch.Tensor) and len(q.shape) == 0
        assert (isinstance(epsilon, torch.Tensor)
                and len(epsilon.shape) == 0 and epsilon > 0.0)
        ctx.save_for_backward(q, epsilon)

        # We solve the optimization problem to find the optimal p
        out = kl_inv(q.item(), epsilon.item(), mode)

        if(out < 0.0):
            out = 10.0**-9

        out = torch.tensor(out, device=q.device)
        ctx.out = out
        ctx.mode = mode
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, epsilon = ctx.saved_tensors
        grad_q = None
        grad_epsilon = None

        # We compute the gradient with respect to q and epsilon
        # (see [1])

        term_1 = (1.0-q)/(1.0-ctx.out)
        term_2 = (q/ctx.out)

        grad_q = torch.log(term_1/term_2)/(term_1-term_2)
        grad_epsilon = (1.0)/(term_1-term_2)

        return grad_output*grad_q, grad_output*grad_epsilon, None

# References:
# [1] Learning Gaussian Processes by Minimizing PAC-Bayesian
#     Generalization Bounds
#     David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch, 2018
###############################################################################

class LogBarrierFunction:
    def __init__(self, t=1.0):
        """
        Initialize the LogBarrierFunction with a specified 't' parameter.
        
        Parameters:
        t (float): Controls the steepness and sensitivity of the barrier.
                   Higher values make the barrier more aggressive.
        """
        self.t = t

    def __call__(self, x):
        """
        Compute the log-barrier for a given constraint 'x'.
        
        Parameters:
        x (torch.Tensor): The constraint variable which should be a scalar tensor.
                          Typically, 'x' would be some function of the model's output
                          that you want to constrain.
        
        Returns:
        torch.Tensor: The value of the log-barrier penalty.
        """
        if x <= -1.0 / (self.t ** 2.0):
            return -(1.0 / self.t) * torch.log(-x)
        else:
            # When x is not within the critical region, use a linear approximation to avoid extreme penalties
            return self.t * x - (1.0 / self.t) * math.log(1 / (self.t ** 2.0)) + (1 / self.t)