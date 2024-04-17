# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
from math import log
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import torch.nn.functional as F

import numpy.linalg as la


def kl(Q, P):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions Q and P.
    Args:
        Q (torch.Tensor): The first probability distribution.
        P (torch.Tensor): The second probability distribution.
    Returns:
        torch.Tensor: The KL divergence between Q and P.
    """
    return F.kl_div(Q.log(), P, reduction='sum')


###############################################################################
def kl_inv(q, epsilon, mode, tol=1e-9, nb_iter_max=1000):
    """
    Solve the optimization problem min{ p in [0, 1] | kl(q||p) <= epsilon }
    or max{ p in [0,1] | kl(q||p) <= epsilon } for q and epsilon fixed using PyTorch

    Parameters
    ----------
    q: float or torch.Tensor
        The parameter q of the kl divergence
    epsilon: float or torch.Tensor
        The upper bound on the kl divergence
    tol: float, optional
        The precision tolerance of the solution
    nb_iter_max: int, optional
        The maximum number of iterations
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

###############################################################################

# References:
# [1] Learning Gaussian Processes by Minimizing PAC-Bayesian
#     Generalization Bounds
#     David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch, 2018



def uniform_distribution(size):
    """
    Generate a uniform distribution of a given size.
    Args:
        size (int): The size of the distribution.
    Returns:
        torch.Tensor: The uniform distribution.
    """
    return torch.full((size,), 1/size)

def risk(preds, targs):
    """
    Calculate the risk of a prediction.

    Parameters:
    preds (array-like): The predicted values.
    targs (array-like): The target values.

    Returns:
    float: The risk value, calculated as 1.0 minus the accuracy score.
    """
    assert(preds.shape == targs.shape)
    return 1.0 - accuracy_score(targs, preds)

def mv_preds(posterior, preds):
    """
    Compute the multiview predictions based on the hyper-posterior probabilities and the predictions in each view.

    Parameters:
    posterior (numpy.ndarray): The posterior probabilities of shape (m,).
    preds (numpy.ndarray): The predictions of shape (n, m), where n is the number of samples and m is the number of views.

    Returns:
    numpy.ndarray: The multiview majority vote predictions of shape (n,).

    Raises:
    AssertionError: If the number of columns in `preds` is not equal to the number of elements in `posterior`.
    """

    m = posterior.shape[0]
    preds = np.transpose(preds)
    assert(preds.shape[1] == m)
    n = preds.shape[0]

    tr = np.min(preds)
    preds -= tr

    results = np.zeros(n)
    for i,pl in enumerate(preds):
        results[i] = np.argmax(np.bincount(pl, weights=posterior))
    return results+tr

def MV_preds(rho, qs, preds):
    rho_qs = qs * rho[:, np.newaxis]
    rho_qs = rho_qs.flatten()
    m = rho_qs.shape[0]
    
    preds = np.concatenate(preds, axis=0)
    preds = np.transpose(preds)
    
    assert(preds.shape[1] == m)
    if rho_qs.sum() != 1:
        print(f"\t\t\t {rho_qs.sum()=}")
    # assert(rho_qs.sum() == 1)
    n = preds.shape[0]

    
    tr = np.min(preds)
    preds -= tr

    results = np.zeros(n)
    for i,pl in enumerate(preds):
        results[i] = np.argmax(np.bincount(pl, weights=rho_qs))
    return results+tr

def oob_risks(preds, targs):
    """
    Calculate the out-of-bag risks and the number of samples for each prediction.

    Args:
        preds (list): A list of tuples containing the predictions and masks for each view.
                      Each tuple should have two arrays: M (mask) and P (prediction).
        targs (array): The true labels for the samples.

    Returns:
        tuple: A tuple containing two arrays: risks and ns.
               - risks: An array of out-of-bag risks for each prediction.
               - ns: An array of the number of samples for each prediction.
    """
    m     = len(preds)
    risks = np.zeros((m,))
    ns    = np.zeros((m,))
    for j, (M, P) in enumerate(preds):
        risks[j] = np.sum(P[M==1]!=targs[M==1])
        ns[j] = np.sum(M)
    return risks, ns


def risks_(preds, targs):
    """
    Calculate the risks of predictions compared to the target values.

    Args:
        preds (numpy.ndarray): The predicted values.
        targs (numpy.ndarray): The target values.

    Returns:
        numpy.ndarray: An array containing the risks for each prediction.

    Raises:
        AssertionError: If the shape of `preds` or `targs` is not as expected.

    """
    assert(len(preds.shape)==2 and len(targs.shape)==1)
    assert(preds.shape[1] == targs.shape[0])
    res = []
    for j in range(preds.shape[0]):
        res.append(np.sum(preds[j]!=targs))
    return np.array(res)

def disagreements(preds):
    """
    Calculates the pairwise disagreements between predictions.

    Args:
        preds (numpy.ndarray): A 2D array of shape (m, n) containing the predictions.

    Returns:
        numpy.ndarray: A 2D array of shape (m, m) containing the pairwise disagreements.
    """
    m, n = preds.shape
    disagreements = np.zeros((m, m))
    for i in range(m):
        for j in range(i, m):
            dis = np.sum(preds[i] != preds[j])
            disagreements[i, j] += dis
            if i != j:
                disagreements[j, i] += dis

    return disagreements

def oob_disagreements(preds):
    m = len(preds)
    disagreements = np.zeros((m,m))
    n2 = np.zeros((m,m))

    for i in range(m):
        (M_i, P_i) = preds[i]
        for j in range(i, m):
            (M_j, P_j) = preds[j]
            M = np.multiply(M_i,M_j)
            disagreements[i,j] = np.sum(P_i[M==1]!=P_j[M==1])
            n2[i,j] = np.sum(M)
            
            if i != j:
                disagreements[j,i] = disagreements[i,j]
                n2[j,i]            = n2[i,j]
    return disagreements, n2    

def tandem_risks(preds, targs):
    """
    Calculate the tandem risks between multiple prediction vectors.

    Args:
        preds (numpy.ndarray): The prediction vectors, where each row represents a prediction vector.
        targs (numpy.ndarray): The target vectors, where each row represents a target vector.

    Returns:
        numpy.ndarray: The tandem risks matrix, where each element represents the tandem risk between two prediction vectors.
    """
    m,n = preds.shape
    tandem_risks = np.zeros((m,m))
    for i in range(m):
        for j in range(i, m):
            tand = np.sum(np.logical_and((preds[i]!=targs), (preds[j]!=targs)))
            tandem_risks[i,j] += tand
            if i != j:
                tandem_risks[j,i] += tand
    return tandem_risks

def oob_tandem_risks(preds, targs):
    m = len(preds)
    tandem_risks  = np.zeros((m,m))
    n2 = np.zeros((m,m))

    for i in range(m):
        (M_i, P_i) = preds[i]
        for j in range(i, m):
            (M_j, P_j) = preds[j]
            M = np.multiply(M_i,M_j)
            tandem_risks[i,j] = np.sum(np.logical_and(P_i[M==1]!=targs[M==1], P_j[M==1]!=targs[M==1]))
            n2[i,j] = np.sum(M)
            
            if i != j:
                tandem_risks[j,i] = tandem_risks[i,j]
                n2[j,i] = n2[i,j]
    
    return tandem_risks, n2