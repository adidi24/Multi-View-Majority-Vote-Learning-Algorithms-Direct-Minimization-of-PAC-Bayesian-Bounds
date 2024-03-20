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

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from .bounds import (optimizeLamb_mv_torch, 
                     PBkl, mv_PBkl, mv_lamb, lamb, 
                     optimizeTND_DIS_mv_torch,
                     tnd, mv_tnd)
from .util import uniform_distribution, mv_preds, risk, kl
import mvpb.util as util


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
    """
    Deep Neural Decision Forests (dNDF) classifier.

    Parameters:
    - depth (int): The depth of the decision trees in the forest.
    - n_in_feature (int): The number of input features.
    - used_feature_rate (float): The rate of randomly selected features used in each decision tree.
    - epochs (int): The number of training epochs.
    - learning_rate (float): The learning rate for the optimizer.

    Methods:
    - fit(X, y): Fit the dNDF model to the training data.
    - predict(X): Predict the labels for the input data.

    """

    def __init__(self, depth, n_in_feature, used_feature_rate, epochs=100, learning_rate=0.001):
        super(DeepNeuralDecisionForests, self).__init__()
        self.depth = depth
        self.n_in_feature = n_in_feature
        self.used_feature_rate = used_feature_rate
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.n_class = len(unique_labels(y))
        self.X_ = torch.from_numpy(X).type(torch.FloatTensor)
        self.y_ = torch.from_numpy(y).type(torch.LongTensor)

        # classifier
        self.model = Tree(self.depth, self.n_in_feature, self.used_feature_rate, self.n_class).to(device)

        # set up DataLoader for training set
        dataset = Dataset(self.X_, self.y_)
        loader = DataLoader(dataset, shuffle=True, batch_size=16)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = loss = F.nll_loss(outputs, labels)
                loss.backward()

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
    
    def __init__(self, nb_estimators, nb_views, depth, used_feature_rate, random_state=42, posterior_rho=None, epochs=100, learning_rate=0.001):
        super(MultiViewBoundsDeepNeuralDecisionForests, self).__init__()
        self.random_state = random_state
        self._prng = check_random_state(self.random_state)
        self.nb_estimators = nb_estimators
        self._estimators_views = None
        self.nb_views = nb_views
        self.classes_ = None
        self.depth = depth
        self.used_feature_rate = used_feature_rate
        self.posterior_rho = posterior_rho if posterior_rho is not None else uniform_distribution(nb_estimators)
        self._abc_pi = uniform_distribution(nb_estimators)  # Initialize the weights with the uniform distribution
        self._OOB = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        for i in range(self.nb_views):
            X[i], y = check_X_y(X[i], y)
        
        
        # Create estimators for each views
        self._estimators_views = [[
            DeepNeuralDecisionForests(depth=self.depth, n_in_feature=X[i].shape[1], used_feature_rate=self.used_feature_rate, epochs=self.epochs, learning_rate=self.learning_rate)
            for _ in range(self.nb_estimators)]for i in range(self.nb_views)]
        
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        for i in range(self.nb_views):
            preds = []
            n = self.X_[i].shape[0]  # Number of samples
            
            for est in self._estimators_views[i]:
                # Sample points for training (w. replacement)
                while True:
                    t_idx = self._prng.randint(n, size=n)
                    t_X = self.X_[i][t_idx]
                    t_Y = self.y_[t_idx]
                    if np.unique(t_Y).shape[0] == len(self.classes_):
                        break
    
                oob_idx = np.delete(np.arange(n), np.unique(t_idx))
                oob_X = self.X_[i][oob_idx]
                
                est.fit(t_X, t_Y)  # Fit this estimator
                oob_P = est.predict(oob_X)  # Predict on OOB
    
                M_est = np.zeros(self.y_.shape)
                P_est = np.zeros(self.y_.shape)
                M_est[oob_idx] = 1
                P_est[oob_idx] = oob_P
                preds.append((M_est, P_est))
                
            print(f'View {i+1}/{self.nb_views} done!')
            self._OOB.append((preds, self.y_))
        return self
    
    def predict_views(self, Xs):
        check_is_fitted(self)
        for i in range(self.nb_views):
            Xs[i] = check_array(Xs[i])
        
        posteriors_qs = [p.data.numpy() for p in self.posterior_Qv]
        
        ys = []
        for v in range(self.nb_views):
            P = [est.predict(Xs[v]).astype(int) for est in self._estimators_views[v]]
            mvtP = mv_preds(posteriors_qs[v], np.array(P))
            ys.append(mvtP)
        return np.array(ys).astype(int)

    def predict_MV(self, Xs, Y=None):
        """
        Return the predicted class labels using majority vote of the
        predictions from each view.
        """
        check_is_fitted(self)
        
        rho = self.posterior_rho.data.numpy()
        
        for i in range(self.nb_views):
            if Y is not None:
                Xs[i], Y = check_X_y(Xs[i], Y)
            else:
                Xs[i] = check_array(Xs[i])
        n_views = len(Xs)

        if n_views != self.nb_views:
            raise ValueError(
                f"Multiview input data must have {self.nb_views} views")
        ys = self.predict_views(Xs)
        mvP = mv_preds(rho, ys)
        # print(f"Xs shapes: {[x.shape for x in Xs]=}\n\n {Y.shape=}\n\n {[y.shape for y in ys]=}\n\n {len(ys)=}\n\n {len(mvP)=}")
        return (mvP, risk(mvP, Y)) if Y is not None else mvP
    
    def  optimize_rho(self, bound, labeled_data=None, incl_oob=True):
        allowed_bounds = {"Lambda", "TND", "DIS"}
        if bound not in allowed_bounds:
            raise Exception(f'Warning, optimize_rho: unknown bound {bound}! expected one of {allowed_bounds}')
        if labeled_data is None and not incl_oob:
            raise Exception('Warning, stats: Missing data! expected labeled_data or incl_oob=True')
        
        check_is_fitted(self)
        
        if bound == 'Lambda':
            risks_views, ns_views = self.risks(labeled_data, incl_oob)
            emp_risks_views = np.divide(risks_views, ns_views, where=ns_views!=0)
            ns_min = torch.tensor(np.min(ns_views))

            posterior_Qv, posterior_rho = optimizeLamb_mv_torch(emp_risks_views, ns_min)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND':
            trisks_views, ns_views = self.tandem_risks(labeled_data, incl_oob)
            dis_views, ns_views = self.disagreements(labeled_data, incl_oob)
            emp_trisks_views = np.divide(trisks_views, ns_views, where=ns_views!=0)
            emp_dis_views = np.divide(dis_views, ns_views, where=ns_views!=0)
            ns_min = torch.tensor(np.min(ns_views))

            posterior_Qv, posterior_rho = optimizeTND_DIS_mv_torch(emp_trisks_views, emp_dis_views, ns_min)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        elif bound == 'DIS':
            raise Exception('Warning, optimize_rho: DIS not implemented yet!')

    def bound(self, bound, labeled_data=None, incl_oob=True):
        if bound not in ['PBkl', 'Lambda', 'TND']:
            raise Exception("Warning, ViewClassifier.bound: Unknown bound!")
        if labeled_data is None and not incl_oob:
            raise Exception('Warning, stats: Missing data! expected labeled_data or incl_oob=True')
        
        m = len(self._OOB[0][0]) if incl_oob else len(labeled_data[0][0])
        v = self.nb_views
        
        # Compute the Kullback-Leibler divergences
        with torch.no_grad():
            prior_pi = uniform_distribution(v)
            prior_Pv = [uniform_distribution(m)]*v
            KL_QPs = [kl(q, p)  for q, p in zip(self.posterior_Qv, prior_Pv)]
            KL_QP = torch.sum(torch.stack(KL_QPs) * self.posterior_rho)
            print(f"{self.posterior_rho=},  {prior_pi=}")
            KL_rhopi = kl(self.posterior_rho, prior_pi)
        
        print(f"{KL_rhopi=},  {KL_QP=}")
        if bound == 'PBkl':
            emp_risks_views, emp_mv_risk, ns = self.mv_risks(labeled_data, incl_oob)
            
            # Compute the PB-kl bound for each view and for the multiview resp.
            pbkl_views = [PBkl(risk, ns, KL_QPs[i].item()) for i, risk in enumerate(emp_risks_views)]
            return (mv_PBkl(emp_mv_risk, ns, KL_QP, KL_rhopi),
                    pbkl_views)
        elif bound == 'Lambda':
            emp_risks_views, emp_mv_risk, ns = self.mv_risks(labeled_data, incl_oob)
            
            # Compute the PB-lambda bound for each view and for the multiview resp.
            lamb_per_view = [lamb(risk, ns, KL_QPs[i].item()) for i, risk in enumerate(emp_risks_views)]
            return (mv_lamb(emp_mv_risk, ns, KL_QP, KL_rhopi),
                    lamb_per_view)
        elif bound == 'TND':
            emp_trisks_views, emp_mv_trisk, ns = self.mv_tandem_risk(labeled_data, incl_oob)
            emp_dis_views, emp_mv_dis, ns = self.mv_disagreement(labeled_data, incl_oob)
            
            # Compute the PB-lambda bound for each view and for the multiview resp.
            tnd_per_view = [tnd(trisk, emp_dis_views[i], ns, KL_QPs[i].item()) for i, trisk in enumerate(emp_trisks_views)]
            return (mv_tnd(emp_mv_trisk, emp_mv_dis, ns, KL_QP, KL_rhopi),
                    tnd_per_view)
        
    def set_posteriors(self, posterior_rho, posterior_Qv):
        self.posterior_rho = posterior_rho
        self.posterior_Qv = posterior_Qv

    def risks(self, data=None, incl_oob=True):
        check_is_fitted(self)
        risks_views = []
        n_views     = []
        for i in range(self.nb_views):
            m = self.nb_estimators
            n = np.zeros((m,))
            risks = np.zeros((m,))
            if incl_oob:
                (preds, targs) = self._OOB[i]
                # preds = [(idx, preds)] * n_estimators
                orisk, on = util.oob_risks(preds, targs)
                n += on
                risks += orisk
                n_views.append(n)
                risks_views.append(risks)
    
            if data is not None:
                assert (len(data) == 2)
                X, Y = data
                P = self.predict_all(X)
    
                n += X[i].shape[0]
                risks += util.risks_(P, Y)

        return risks_views, n_views
    
    def mv_risks(self, labeled_data=None, incl_oob=True):
        risks_views, ns_views = self.risks(labeled_data, incl_oob)
        emp_risks_views = np.divide(risks_views, ns_views, where=ns_views!=0)
        emp_rv = []
        for q, rv in zip(self.posterior_Qv, emp_risks_views):
            emp_rv.append(np.average(rv, weights=q.detach().numpy(), axis=0))

        emp_mv_risk = np.average(emp_rv, weights=self.posterior_rho.detach().numpy(), axis=0)
        return np.array(emp_rv), emp_mv_risk, np.min(ns_views)

    
    def mv_tandem_risk(self, labeled_data=None, incl_oob=True):
        trsk, n2 = self.tandem_risks(labeled_data, incl_oob)
        trsk = np.divide(trsk, n2, where=n2!=0)
        
        emp_tnd_rv = []
        for q, rv in zip(self.posterior_Qv, trsk):
            qv = q.detach().numpy()
            emp_tnd_rv.append(np.average(np.average(rv, weights=qv, axis=0), 
                                         weights=qv))
            

        emp_tnd_rv = np.array(emp_tnd_rv)
        mv_trisks = np.outer(emp_tnd_rv, emp_tnd_rv)
        emp_mv_tnd_risk = np.average(
            np.average(mv_trisks, weights=self.posterior_rho.detach().numpy(), axis=0),
            weights=self.posterior_rho.detach().numpy())

        return emp_tnd_rv, emp_mv_tnd_risk, np.min(n2)

    def tandem_risks(self, data=None, incl_oob=True):
        check_is_fitted(self)
        tandem_risks_views = []
        n_views     = []
        for i in range(self.nb_views):
            m = self.nb_estimators
            n2 = np.zeros((m, m))
            tandem_risks = np.zeros((m, m))

            if incl_oob:
                (preds, targs) = self._OOB[i]
                # preds = [(idx, preds)] * n_estimators
                otand, on2 = util.oob_tandem_risks(preds, targs)
                n2 += on2
                tandem_risks += otand
                n_views.append(n2)
                tandem_risks_views.append(tandem_risks)

            if data is not None:
                assert (len(data) == 2)
                X, Y = data
                P = self.predict_all(X)

                n2 += X.shape[0]
                tandem_risks += util.tandem_risks(P, Y)

        return tandem_risks_views, n_views
    
    # Returns the disagreement
    def mv_disagreement(self, labeled_data=None, incl_oob=True):
        dis, n2 = self.disagreements(labeled_data, incl_oob)
        dis = np.divide(dis, n2, where=n2!=0)
        
        emp_dis_rv = []
        for q, rv in zip(self.posterior_Qv, dis):
            qv = q.detach().numpy()
            emp_dis_rv.append(np.average(np.average(rv, weights=qv, axis=0), 
                                         weights=qv))
            

        emp_dis_rv = np.array(emp_dis_rv)
        mv_dis = np.outer(emp_dis_rv, emp_dis_rv)
        emp_mv_dis_risk = np.average(
            np.average(mv_dis, weights=self.posterior_rho.detach().numpy(), axis=0),
            weights=self.posterior_rho.detach().numpy())

        return emp_dis_rv, emp_mv_dis_risk, np.min(n2)

    def disagreements(self, data=None, incl_oob=True):
        check_is_fitted(self)
        disagreements_views = []
        n_views     = []
        for i in range(self.nb_views):
            m = self.nb_estimators
            n2 = np.zeros((m, m))
            disagreements = np.zeros((m, m))

            if incl_oob:
                (preds, Y) = self._OOB[i]
                # preds = [(idx, preds)] * n_estimators
                odis, on2 = util.oob_disagreements(preds)
                n2 += on2
                disagreements += odis
                n_views.append(n2)
                disagreements_views.append(disagreements)

            if data is not None:
                X = data
                P = self.predict_all(X)

                n2 += X.shape[0]
                disagreements += util.disagreements(P)

        return disagreements_views, n_views