# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score


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

    Args:
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

    Args:
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
    """
    Compute the Multi-view predictions using the given weights and predictions.

    Args:
        rho (numpy.ndarray):  The hyper-posterior probabilities of shape (nb_views,).
        qs (numpy.ndarray): The posterior probabilities of shape (nb_views, nb_estimators).
        preds (list): The list of prediction arrays for each view.

    Returns:
        numpy.ndarray: The Multi-view predictions.

    Raises:
        AssertionError: If the shape of the predictions array is not valid.

    """
    rho_qs = qs * rho[:, np.newaxis]
    rho_qs = rho_qs.flatten()
    m = rho_qs.shape[0]
    preds = np.concatenate(preds, axis=0)
    
    preds = np.transpose(preds)
    
    assert(preds.shape[1] == m)
    # if rho_qs.sum() != 1:
    #     print(f"\t\t\t {rho_qs.sum()=}")
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

def multiview_risks_(preds, targs):
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
    assert(len(preds.shape)==3 and len(targs.shape)==1)
    assert(preds.shape[2] == targs.shape[0])
    num_views, num_estimators, _ = preds.shape
    risks = np.zeros((num_views, num_estimators))
    # res = []
    for i in range(num_views):
        for j in range(num_estimators):
            risks[i, j] = np.sum(preds[i, j] != targs)
    return risks

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

def multiview_disagreements(preds):
    """
    Calculates the multiview pairwise disagreements between predictions.

    Args:
        preds (numpy.ndarray): A 3D array of shape (num_views, num_estimators_per_view, num_samples)
                                containing the predictions for each view.

    Returns:
        numpy.ndarray: A 3D array of shape (num_views, num_views, num_estimators_per_view) containing
                        the multiview pairwise disagreements.
    """
    num_views, num_estimators, _ = preds.shape
    disagreements = np.zeros((num_views, num_views, num_estimators, num_estimators))
    
    # Compute disagreements between each pair of views
    for i in range(num_views):
        for j in range(i, num_views):
            for k in range(num_estimators):
                for l in range(k, num_estimators):
                    dis = np.sum(preds[i, k] != preds[j, l])
                    disagreements[i, j, k, l] = dis
                    if k != l:
                        disagreements[i, j, l, k] = dis
            if i != j:
                disagreements[j, i, :, :] = disagreements[i, j, :, :]
                

    return disagreements

def multiview_oob_disagreements(preds):
    """
    Calculates the multiview out-of-bag pairwise disagreements between predictions.

    Args:
        preds (list): A list of length m containing tuples (M, P) for each view, where:
                      - M is the out-of-bag mask matrix of shape (n, n) indicating which samples were out-of-bag
                      - P is the predictions matrix of shape (n, n_estimators) containing the predictions

    Returns:
        tuple: A tuple (disagreements, n2) where:
               - disagreements is a 3D array of shape (m, m, n_estimators) containing the multiview
                 pairwise disagreements for each estimator
               - n2 is a 3D array of shape (m, m, n_estimators) containing the counts of out-of-bag samples
    """
    m = len(preds)
    n_estimators = preds[0][1].shape[1]  # Assuming all views have the same number of estimators
    disagreements = np.zeros((m, m, n_estimators,n_estimators))
    n2 = np.zeros((m, m, n_estimators,n_estimators))
    
    for i in range(m):
        M_i, P_i = preds[i]
        for j in range(i, m):
            M_j, P_j = preds[j]
            M = np.multiply(M_i, M_j)
            for k in range(n_estimators):
                disagreements[i, j, k] = np.sum(P_i[M == 1, k] != P_j[M == 1, k])
                n2[i, j, k] = np.sum(M[:, k])

                if i != j:
                    disagreements[j, i, k] = disagreements[i, j, k]
                    n2[j, i, k] = n2[i, j, k]

    return disagreements, n2


def joint_errors(preds, targs):
    """
    Calculate the joint errors between multiple prediction vectors.

    Args:
        preds (numpy.ndarray): The prediction vectors, where each row represents a prediction vector.
        targs (numpy.ndarray): The target vectors, where each row represents a target vector.

    Returns:
        numpy.ndarray: The joint errors matrix, where each element represents the joint error between two prediction vectors.
    """
    m,n = preds.shape
    joint_errors = np.zeros((m,m))
    for i in range(m):
        for j in range(i, m):
            e = np.sum(np.logical_and((preds[i]!=targs), (preds[j]!=targs)))
            joint_errors[i,j] += e
            if i != j:
                joint_errors[j,i] += e
    return joint_errors

def oob_joint_errors(preds, targs):
    m = len(preds)
    joint_errors  = np.zeros((m,m))
    n2 = np.zeros((m,m))

    for i in range(m):
        (M_i, P_i) = preds[i]
        for j in range(i, m):
            (M_j, P_j) = preds[j]
            M = np.multiply(M_i,M_j)
            joint_errors[i,j] = np.sum(np.logical_and(P_i[M==1]!=targs[M==1], P_j[M==1]!=targs[M==1]))
            n2[i,j] = np.sum(M)
            
            if i != j:
                joint_errors[j,i] = joint_errors[i,j]
                n2[j,i] = n2[i,j]
    
    return joint_errors, n2

def multiview_joint_errors(preds, targs):
    """
    Calculate the multiview joint errors between multiple prediction vectors.

    Args:
        preds (numpy.ndarray): The prediction vectors, where each row represents a prediction vector.
        targs (numpy.ndarray): The target vectors, where each row represents a target vector.

    Returns:
        numpy.ndarray: The joint errors matrix, where each element represents the joint error between two prediction vectors.
    """
    num_views, num_estimators, _ = preds.shape
    joint_errors = np.zeros((num_views, num_views, num_estimators, num_estimators))
    
    # Compute disagreements between each pair of views
    for i in range(num_views):
        for j in range(i, num_views):
            for k in range(num_estimators):
                for l in range(k, num_estimators):
                    e = np.sum(np.logical_and((preds[i, k]!=targs), (preds[j, l]!=targs)))
                    joint_errors[i, j, k, l] = e
                    if k != l:
                        joint_errors[i, j, l, k] = e
            if i != j:
                joint_errors[j, i, :, :] = joint_errors[i, j, :, :]
    return joint_errors


