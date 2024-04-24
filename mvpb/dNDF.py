# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 07:30:57 2024

@author: mehdihennequin
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import mvpb.bounds_alpha1_KL as bkl
import mvpb.bounds_renyi as br
from .util import uniform_distribution, mv_preds, risk, kl
from .util import renyi_divergence as rd
import mvpb.util as util

from .deepTree import DeepNeuralDecisionForests
from .tree import Tree


    


class MultiViewBoundsDeepNeuralDecisionForests(BaseEstimator, ClassifierMixin):
    
    def __init__(self, nb_estimators, nb_views, depth, used_feature_rate, random_state=42, posterior_rho=None, epochs=100, learning_rate=0.001, use_dndf=False):
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
        self.lamb = None
        self.lamb_tnd = None
        self.lamb1_tnd_dis = None
        self.lamb2_tnd_dis = None
        self.lamb_dis = None
        self.gamma_dis = None
        self.use_dndf = use_dndf
        
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        for i in range(self.nb_views):
            X[i], y = check_X_y(X[i], y)
        
        
        # Create estimators for each views
        if self.use_dndf:
            self._estimators_views = [[
                DeepNeuralDecisionForests(depth=self.depth, n_in_feature=X[i].shape[1], used_feature_rate=self.used_feature_rate, epochs=self.epochs, learning_rate=self.learning_rate)
                for _ in range(self.nb_estimators)]for i in range(self.nb_views)]
        else:
            self._estimators_views = [[
                Tree(rand=self._prng, max_features="sqrt", max_depth=self.depth)
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
                
                est = est.fit(t_X, t_Y)  # Fit this estimator
                oob_P = est.predict(oob_X)  # Predict on OOB
    
                M_est = np.zeros(self.y_.shape)
                P_est = np.zeros(self.y_.shape)
                M_est[oob_idx] = 1
                P_est[oob_idx] = oob_P
                preds.append((M_est, P_est))
                
            # print(f'View {i+1}/{self.nb_views} done!')
            self._OOB.append((preds, self.y_))
        return self
    
    def predict_views(self, Xs, Y=None):
        check_is_fitted(self)
        
        posteriors_qs = [p.data.numpy() for p in self.posterior_Qv]
        
        ys = []
        v_risks = []
        for v in range(self.nb_views):
            P = [est.predict(Xs[v]).astype(int) for est in self._estimators_views[v]]
            mvtP = util.mv_preds(posteriors_qs[v], np.array(P))
            ys.append(mvtP)
            if Y is not None:
                v_risks.append(util.risk(mvtP, Y))
        return np.array(ys).astype(int), np.array(v_risks)

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
        ys, v_risks = self.predict_views(Xs, Y)
        mvP = util.mv_preds(rho, ys)
        # print(f"Xs shapes: {[x.shape for x in Xs]=}\n\n {Y.shape=}\n\n {[y.shape for y in ys]=}\n\n {len(ys)=}\n\n {len(mvP)=}")
        return (mvP, util.risk(mvP, Y), v_risks) if Y is not None else mvP
    
    def  optimize_rho(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, max_iter=1000, optimise_lambda_gamma=False, alpha=1):
        allowed_bounds = {'PBkl', 'Lambda', 'TND_DIS', 'TND', 'DIS'}
        if bound not in allowed_bounds:
            raise Exception(f'Warning, optimize_rho: unknown bound {bound}! expected one of {allowed_bounds}')
        if labeled_data is None and not incl_oob:
            raise Exception('Warning, stats: Missing data! expected labeled_data or incl_oob=True')

        check_is_fitted(self)
        
        ulX =  None
        if unlabeled_data is not None:
            if labeled_data is None:
                ulX = unlabeled_data  
            else:
                ulX = []
                for i in range(self.nb_views):
                    ulX.append(np.concatenate((labeled_data[0][i], unlabeled_data[i]), axis=0))
        else:
            if labeled_data is not None:
                ulX = labeled_data[0]
            
        
        if bound == 'Lambda':
            risks_views, ns_views = self.risks(labeled_data, incl_oob)
            emp_risks_views = np.divide(risks_views, ns_views, where=ns_views!=0)
            ns_min = torch.tensor(np.min(ns_views))

            if alpha == 1:
                posterior_Qv, posterior_rho, lamb = bkl.optimizeLamb_mv_torch(emp_risks_views, ns_min, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb = br.optimizeLamb_mv_torch(emp_risks_views, ns_min, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb = lamb
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND_DIS':
            trisks_views, ns_views_t = self.tandem_risks(labeled_data, incl_oob)
            dis_views, ns_views_d = self.disagreements(ulX, incl_oob)
            emp_trisks_views = np.divide(trisks_views, ns_views_t, where=ns_views_t!=0)
            emp_dis_views = np.divide(dis_views, ns_views_d, where=ns_views_d!=0)
            nt = torch.tensor(np.min(ns_views_t))
            nd = torch.tensor(np.min(ns_views_d))

            if alpha == 1:
                posterior_Qv, posterior_rho, lamb1_tnd_dis, lamb2_tnd_dis = bkl.optimizeTND_DIS_mv_torch(emp_trisks_views, emp_dis_views, nt, nd, optimise_lambdas=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb1_tnd_dis, lamb2_tnd_dis = br.optimizeTND_DIS_mv_torch(emp_trisks_views, emp_dis_views, nt, nd, optimise_lambdas=optimise_lambda_gamma, alpha=alpha)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            # print(f"{lamb1_tnd_dis=}, {lamb2_tnd_dis=}")
            self.lamb1_tnd_dis = lamb1_tnd_dis
            self.lamb2_tnd_dis = lamb2_tnd_dis
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND':
            trisks_views, ns_views = self.tandem_risks(labeled_data, incl_oob)
            emp_trisks_views = np.divide(trisks_views, ns_views, where=ns_views!=0)
            ns_min = torch.tensor(np.min(ns_views))

            if alpha == 1:
                posterior_Qv, posterior_rho, lamb_tnd = bkl.optimizeTND_mv_torch(emp_trisks_views, ns_min, optimise_lambda=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb_tnd = br.optimizeTND_mv_torch(emp_trisks_views, ns_min, optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_tnd=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb_tnd = lamb_tnd
            return posterior_Qv, posterior_rho
        elif bound == 'DIS':
            risks_views, ns_views_g = self.risks(labeled_data, incl_oob)
            dis_views, ns_views_d = self.disagreements(ulX, incl_oob)
            emp_risks_views = np.divide(risks_views, ns_views_g, where=ns_views_g!=0)
            emp_dis_views = np.divide(dis_views, ns_views_d, where=ns_views_d!=0)
            ng = torch.tensor(np.min(ns_views_g))
            nd = torch.tensor(np.min(ns_views_d))

            if alpha == 1:
                posterior_Qv, posterior_rho, lamb_dis, gamma_dis = bkl.optimizeDIS_mv_torch(emp_risks_views, emp_dis_views, ng, nd, optimise_lambda_gamma=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb_dis, gamma_dis = br.optimizeDIS_mv_torch(emp_risks_views, emp_dis_views, ng, nd, optimise_lambda_gamma=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_dis=}, {gamma_dis=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb_dis = lamb_dis
            self.gamma_dis = gamma_dis
            return posterior_Qv, posterior_rho
        else:
            raise Exception(f'Warning, optimize_rho: unknown bound {bound}! expected one of {allowed_bounds}')

    def bound(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, alpha=1.0):
        if bound not in ['PBkl', 'Lambda', 'TND_DIS', 'TND', 'DIS']:
            raise Exception("Warning, ViewClassifier.bound: Unknown bound!")
        
        m = self.nb_estimators
        v = self.nb_views
        
        ulX =  None
        if unlabeled_data is not None:
            if labeled_data is None:
                ulX = unlabeled_data  
            else:
                ulX = []
                for i in range(self.nb_views):
                    ulX.append(np.concatenate((labeled_data[0][i], unlabeled_data[i]), axis=0))
        else:
            if labeled_data is not None:
                ulX = labeled_data[0]
                
        
        # Compute the Kullback-Leibler divergences
        with torch.no_grad():
            prior_pi = uniform_distribution(v)
            prior_Pv = [uniform_distribution(m)]*v
            if alpha==1:
                KL_QPs = [kl(q, p)  for q, p in zip(self.posterior_Qv, prior_Pv)]
                KL_QP = torch.sum(torch.stack(KL_QPs) * self.posterior_rho)
                # print(f"{self.posterior_rho=},  {prior_pi=}")
                KL_rhopi = kl(self.posterior_rho, prior_pi)
            else:
                RD_QPs = [rd(q, p, alpha)  for q, p in zip(self.posterior_Qv, prior_Pv)]
                RD_QP = torch.sum(torch.stack(RD_QPs) * self.posterior_rho)
                RD_rhopi = rd(self.posterior_rho, prior_pi, alpha)
        
        # print(f"{KL_rhopi=},  {KL_QP=}")
        if bound == 'PBkl':
            emp_risks_views, emp_mv_risk, ns = self.mv_risks(labeled_data, incl_oob)
            
            # Compute the PB-kl bound for each view and for the multiview resp.
            if alpha==1:
                pbkl_views = [bkl.PBkl(risk, ns, KL_QPs[i].item()) for i, risk in enumerate(emp_risks_views)]
                return (bkl.mv_PBkl(emp_mv_risk, ns, KL_QP, KL_rhopi),
                    pbkl_views)
            else:
                raise Exception("Warning, ViewClassifier.bound: PBkl not implemented for alpha != 1")
            
        elif bound == 'Lambda':
            emp_risks_views, emp_mv_risk, ns = self.mv_risks(labeled_data, incl_oob)
            
            # Compute the PB-lambda bound for each view and for the multiview resp.
            if alpha==1:
                lamb_per_view = [bkl.lamb(risk, ns, KL_QPs[i].item()) for i, risk in enumerate(emp_risks_views)]
                return (bkl.mv_lamb(emp_mv_risk, ns, KL_QP.item(), KL_rhopi.item(), lamb=self.lamb),
                        lamb_per_view)
            else:
                lamb_per_view = [br.lamb(risk, ns, RD_QPs[i].item(), alpha) for i, risk in enumerate(emp_risks_views)]
                return (br.mv_lamb(emp_mv_risk, ns, RD_QP.item(), RD_rhopi.item(), lamb=self.lamb),
                        lamb_per_view)
        elif bound == 'TND_DIS':
            emp_trisks_views, emp_mv_trisk, nt = self.mv_tandem_risk(labeled_data, incl_oob)
            emp_dis_views, emp_mv_dis, nd = self.mv_disagreement(ulX, incl_oob)
            
            # Compute the TND_DIS bound for each view and for the multiview resp.
            if alpha==1:
                tnd_per_view = [bkl.tnd_dis(trisk, emp_dis_views[i], nt, nd, KL_QPs[i].item()) for i, trisk in enumerate(emp_trisks_views)]
                return (bkl.mv_tnd_dis(emp_mv_trisk, 
                                emp_mv_dis, nt, nd, 
                                KL_QP.item(), KL_rhopi.item(), 
                                lamb1=self.lamb1_tnd_dis, lamb2=self.lamb2_tnd_dis),
                        tnd_per_view)
            else:
                tnd_per_view = [br.tnd_dis(trisk, emp_dis_views[i], nt, nd, RD_QPs[i].item(), alpha) for i, trisk in enumerate(emp_trisks_views)]
                return (br.mv_tnd_dis(emp_mv_trisk, emp_mv_dis, nt, nd, 
                                RD_QP.item(), RD_rhopi.item(), 
                                lamb1=self.lamb1_tnd_dis, lamb2=self.lamb2_tnd_dis),
                        tnd_per_view)
        elif bound == 'TND':
            emp_trisks_views, emp_mv_trisk, n = self.mv_tandem_risk(labeled_data, incl_oob)
            
            # Compute the TND bound for each view and for the multiview resp.
            if alpha==1:
                tnd_per_view = [bkl.tnd(trisk, n, KL_QPs[i].item()) for i, trisk in enumerate(emp_trisks_views)]
                return (bkl.mv_tnd(emp_mv_trisk, n, KL_QP.item(), KL_rhopi.item(), lamb=self.lamb_tnd),
                        tnd_per_view)
            else:
                tnd_per_view = [br.tnd(trisk, n, RD_QPs[i].item(), alpha) for i, trisk in enumerate(emp_trisks_views)]
                return (br.mv_tnd(emp_mv_trisk, n, RD_QP.item(), RD_rhopi.item(), lamb=self.lamb_tnd),
                        tnd_per_view)
        elif bound == 'DIS':
            emp_risks_views, emp_mv_risk, ng = self.mv_risks(labeled_data, incl_oob)
            emp_dis_views, emp_mv_dis, nd = self.mv_disagreement(ulX, incl_oob)
            
            # Compute the DIS bound for each view and for the multiview resp.
            if alpha==1:
                tnd_per_view = [bkl.dis(grisk, emp_dis_views[i], ng, nd, KL_QPs[i].item()) for i, grisk in enumerate(emp_risks_views)]
                return (bkl.mv_dis(emp_mv_risk, emp_mv_dis, ng, nd,
                            KL_QP.item(), KL_rhopi.item(),
                            lamb=self.lamb_dis, gamma=self.gamma_dis),
                        tnd_per_view)
            else:
                tnd_per_view = [br.dis(grisk, emp_dis_views[i], ng, nd, RD_QPs[i].item(), alpha) for i, grisk in enumerate(emp_risks_views)]
                return (br.mv_dis(emp_mv_risk, emp_mv_dis, ng, nd,
                            RD_QP.item(), RD_rhopi.item(),
                            lamb=self.lamb_dis, gamma=self.gamma_dis),
                        tnd_per_view)
        
    def set_posteriors(self, posterior_rho, posterior_Qv):
        self.posterior_rho = posterior_rho
        self.posterior_Qv = posterior_Qv
    
    def clear_posteriors(self):
        self.posterior_rho = None
        self.posterior_Qv = None

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
    
            if data is not None:
                assert (len(data) == 2)
                X, Y = data
                P = np.array([est.predict(X[i]).astype(int) for est in self._estimators_views[i]])
    
                n += X[i].shape[0]
                risks += util.risks_(P, Y)
            n_views.append(n)
            risks_views.append(risks)

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

            if data is not None:
                assert (len(data) == 2)
                X, Y = data
                P = np.array([est.predict(X[i]).astype(int) for est in self._estimators_views[i]])

                n2 += X[i].shape[0]
                tandem_risks += util.tandem_risks(P, Y)
            n_views.append(n2)
            tandem_risks_views.append(tandem_risks)

        return tandem_risks_views, n_views
    
    # Returns the disagreement
    def mv_disagreement(self, unlabeled_data=None, incl_oob=True):
        dis, n2 = self.disagreements(unlabeled_data, incl_oob)
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

    def disagreements(self, unlabeled_data=None, incl_oob=True):
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

            if unlabeled_data is not None:
                X = unlabeled_data[i]
                P = np.array([est.predict(X).astype(int) for est in self._estimators_views[i]])

                n2 += X.shape[0]
                disagreements += util.disagreements(P)
            n_views.append(n2)
            disagreements_views.append(disagreements)

        return disagreements_views, n_views