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

from math import log, sqrt
import numpy as np
from scipy.special import gammaln
from scipy import optimize

from .util import kl

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
    return sum([ q*log(q/p) if q > 0. else 0. for q,p in zip(Q,P) ])


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