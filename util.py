#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

import numpy as np
import numpy.linalg as la
from math import log
from sklearn.metrics import accuracy_score


def kl(Q,P):
    return F.kl_div(Q.log(), P, reduction='sum')


def uniform_distribution(size):
    return torch.full((size,), 1/size)