# -*- coding: utf-8 -*-

import torch
import numpy as np
from math import log
from sklearn.metrics import accuracy_score

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
    return 1.0 - accuracy_score(preds, targs)

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