import torch
import numpy as np
from math import log
from sklearn.metrics import accuracy_score

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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