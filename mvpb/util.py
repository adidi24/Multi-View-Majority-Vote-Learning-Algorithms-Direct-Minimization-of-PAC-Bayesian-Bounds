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