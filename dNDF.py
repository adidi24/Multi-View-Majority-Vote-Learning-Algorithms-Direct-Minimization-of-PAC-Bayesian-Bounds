#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 07:30:57 2024

@author: mehdihennequin
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def uniform_distribution(m):
    return np.full((m,), 1.0/m)

class Dataset(Dataset):

  def __init__(self, Data, labels):
        'Initialization'
        self.Data = Data
        self.labels = labels
        

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, idx):
        'Generates one sample of data'


        # Load data and get label
        X = self.Data[idx]
        y = self.labels[idx]

        return X, y
    

class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class

        # used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = torch.eye(n_in_feature)
        using_idx = torch.randperm(n_in_feature)[:n_used_feature]
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = nn.Parameter(self.feature_mask, requires_grad=False)
        # leaf label distribution
        self.pi = nn.Parameter(torch.rand(self.n_leaf, n_class), requires_grad=True)

        # decision
        self.decision = nn.Sequential(
            nn.Linear(n_used_feature, self.n_leaf),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x, self.feature_mask)  # ->[batch_size, n_used_feature]
        decision = self.decision(feats)  # ->[batch_size, n_leaf]

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # -> [batch_size, n_leaf, 2]

        batch_size = x.size()[0]
        _mu = x.data.new(batch_size, 1, 1).fill_(1.)

        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]
            _mu = _mu * _decision
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        # Calculating class probabilities
        pi = F.softmax(self.pi, dim=-1)
        class_probs = torch.matmul(mu, pi)

        # Returning log probabilities for nll_loss
        log_probs = F.log_softmax(class_probs, dim=1)

        return log_probs

    
    def get_pi(self):
        return F.softmax(self.pi, dim=-1)


    def cal_prob(self,mu,pi):
        """

        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu,pi)
        return p


    def update_pi(self,new_pi):
        self.pi.data=new_pi
        
        
class DeepNeuralDecisionForests(BaseEstimator, ClassifierMixin):
    
    def __init__(self, depth,n_in_feature,used_feature_rate, epochs=100, learning_rate = 0.001):
        self.depth             = depth
        self.n_in_feature      = n_in_feature
        self.used_feature_rate = used_feature_rate
        self.epochs            = epochs
        self.learning_rate     = learning_rate
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.n_class = len(unique_labels(y))
        self.X_ = torch.from_numpy(X).type(torch.FloatTensor)
        self.y_ = torch.from_numpy(y).type(torch.LongTensor)
        
        #classifier
        self.model = Tree(self.depth,self.n_in_feature,self.used_feature_rate,self.n_class)
        # enumerate epochs
        # set up DataLoader for training set
        dataset = Dataset(self.X_, self.y_)
        loader = DataLoader(dataset, shuffle=True, batch_size=16)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        for epoch in range(self.epochs): 
            #print('EPOCH {}:'.format(n_epochs + 1))
        
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            for batch_idx, data in enumerate(loader):
                # Every data instance is an input + label pair
                
                inputs, labels = data
                                # Zero your gradients for every batch!
                optimizer.zero_grad()
        
                # Make predictions for this batch
                outputs = self.model(inputs)
        
                # Compute the loss and its gradients
                loss = loss = F.nll_loss(outputs, labels)
                loss.backward()
        
                # Adjust learning weights
                optimizer.step()
            
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.from_numpy(X).type(torch.FloatTensor)
    
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted_labels = torch.argmax(outputs, dim=1)
    
        return predicted_labels.cpu().numpy()
    
    

class MultiViewBoundsDeepNeuralDecisionForests(BaseEstimator, ClassifierMixin):
    
    def __init__(self, nb_estimators, nb_views, depth, n_in_feature, used_feature_rate, random_state=42, posterior=None):
        self.nb_estimators = nb_estimators
        self.nb_views = nb_views
        self.depth = depth
        self.n_in_feature = n_in_feature
        self.used_feature_rate = used_feature_rate
        self.random_state = random_state
        self.posterior = posterior if posterior is not None else uniform_distribution(nb_estimators)
        self._abc_pi = uniform_distribution(nb_estimators)  # Initialize the weights with the uniform distribution
        self._OOB = None  # Some fitting stats
        
    def fit(self, X, y):
        
        self._prng = check_random_state(self.random_state)
        # Create estimators
        self._estimators = [
            DeepNeuralDecisionForests(depth=self.depth, n_in_feature=self.n_in_feature, used_feature_rate=self.used_feature_rate)
            for _ in range(self.nb_estimators)
        ]
        
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        preds = []
        n = X.shape[0]  # Number of samples
        for est in self._estimators:
            # Sample points for training (w. replacement)
            while True:
                t_idx = self._prng.randint(n, size=n)
                t_X = self.X_[t_idx]
                t_Y = self.y_[t_idx]
                if np.unique(t_Y).shape[0] == len(self.classes_):
                    break

            oob_idx = np.delete(np.arange(n), np.unique(t_idx))
            oob_X = X[oob_idx]
            
            est.fit(t_X, t_Y)  # Fit this estimator
            oob_P = est.predict(oob_X)  # Predict on OOB

            M_est = np.zeros(self.y_.shape)
            P_est = np.zeros(self.y_.shape)
            M_est[oob_idx] = 1
            P_est[oob_idx] = oob_P
            preds.append((M_est, P_est))

        self._OOB = (preds, self.y_)
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


    def risk(self, data=None):
        if data is None and self._sample_mode is None:
            util.warn('Warning, MVBase.risk: No OOB data!')
            return 1.0
        if data is None:
            #### WARNING: not implemented, not used TODO.
            (preds, targs) = self._OOB
            return 1.0  # util.oob_estimate(self._rho, preds, targs)
        else:
            (X, Y) = data
            P = self.predict_all(X)
            return util.mv_risk(self._rho, P, Y)
    
    
