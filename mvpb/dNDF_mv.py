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
        self.posterior_rho = posterior_rho if posterior_rho is not None else uniform_distribution(nb_views)
        self.posterior_Qv = [uniform_distribution(nb_estimators) for _ in range(nb_views)]
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
        
        posteriors_qs = [p.cpu().data.numpy() for p in self.posterior_Qv]
        
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
        
        rho = self.posterior_rho.cpu().data.numpy()
        posteriors_qs = [p.cpu().data for p in self.posterior_Qv]
        
        for i in range(self.nb_views):
            if Y is not None:
                Xs[i], Y = check_X_y(Xs[i], Y)
            else:
                Xs[i] = check_array(Xs[i])
        n_views = len(Xs)

        if n_views != self.nb_views:
            raise ValueError(
                f"Multiview input data must have {self.nb_views} views")
        
        mys = []
        for v in range(self.nb_views):
            mys.append([est.predict(Xs[v]).astype(int) for est in self._estimators_views[v]])
        mvP = util.MV_preds(rho, np.array(posteriors_qs), mys)
            
        # ys, v_risks = self.predict_views(Xs, Y)
        # mvP2 = util.mv_preds(rho, ys)
        
        # assert mvP.shape[0] == mvP2.shape[0]
        # for i in range(mvP.shape[0]):
        #     assert mvP[i] == mvP2[i]
        # print(f"risk mvP2:{util.risk(mvP2, Y)}\n risk mvP{util.risk(mvP, Y)}")
        # print(f"Xs shapes: {[x.shape for x in Xs]=}\n\n {Y.shape=}\n\n {[y.shape for y in ys]=}\n\n {len(ys)=}\n\n {len(mvP)=}")
        return (mvP, util.risk(mvP, Y)) if Y is not None else mvP
    
    def  optimize_rho(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, max_iter=1000, optimise_lambda_gamma=False, alpha=1):
        allowed_bounds = {'Lambda', 'TND_DIS', 'TND', 'DIS'}
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
                posterior_Qv, posterior_rho, lamb = bkl.optimizeLamb_mv_torch(emp_risks_views, ns_min, device, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb = br.optimizeLamb_mv_torch(emp_risks_views, ns_min, device, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb = lamb
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND_DIS':
            trisks_views, ns_views_t = self.multiview_tandem_risks(labeled_data, incl_oob)
            dis_views, ns_views_d = self.multiview_disagreements(ulX, incl_oob)
            emp_trisks_views = np.divide(trisks_views, ns_views_t, where=ns_views_t!=0)
            emp_dis_views = np.divide(dis_views, ns_views_d, where=ns_views_d!=0)
            nt = torch.tensor(np.min(ns_views_t))
            nd = torch.tensor(np.min(ns_views_d))

            if alpha == 1:
                posterior_Qv, posterior_rho, lamb1_tnd_dis, lamb2_tnd_dis = bkl.optimizeTND_DIS_mv_torch(emp_trisks_views, emp_dis_views, nt, nd, device, optimise_lambdas=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb1_tnd_dis, lamb2_tnd_dis = br.optimizeTND_DIS_mv_torch(emp_trisks_views, emp_dis_views, nt, nd, device, optimise_lambdas=optimise_lambda_gamma, alpha=alpha)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            # print(f"{lamb1_tnd_dis=}, {lamb2_tnd_dis=}")
            self.lamb1_tnd_dis = lamb1_tnd_dis
            self.lamb2_tnd_dis = lamb2_tnd_dis
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND':
            trisks_views, ns_views = self.multiview_tandem_risks(labeled_data, incl_oob)
            emp_trisks_views = np.divide(trisks_views, ns_views, where=ns_views!=0)
            ns_min = torch.tensor(np.min(ns_views))

            if alpha == 1:
                posterior_Qv, posterior_rho, lamb_tnd = bkl.optimizeTND_mv_torch(emp_trisks_views, ns_min, device, optimise_lambda=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb_tnd = br.optimizeTND_mv_torch(emp_trisks_views, ns_min, device, optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_tnd=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb_tnd = lamb_tnd
            return posterior_Qv, posterior_rho
        elif bound == 'DIS':
            risks_views, ns_views_g = self.risks(labeled_data, incl_oob)
            dis_views, ns_views_d = self.multiview_disagreements(ulX, incl_oob)
            emp_risks_views = np.divide(risks_views, ns_views_g, where=ns_views_g!=0)
            emp_dis_views = np.divide(dis_views, ns_views_d, where=ns_views_d!=0)
            ng = torch.tensor(np.min(ns_views_g))
            nd = torch.tensor(np.min(ns_views_d))

            if alpha == 1:
                posterior_Qv, posterior_rho, lamb_dis, gamma_dis = bkl.optimizeDIS_mv_torch(emp_risks_views, emp_dis_views, ng, nd, device, optimise_lambda_gamma=optimise_lambda_gamma)
            else:
                posterior_Qv, posterior_rho, lamb_dis, gamma_dis = br.optimizeDIS_mv_torch(emp_risks_views, emp_dis_views, ng, nd, device, optimise_lambda_gamma=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_dis=}, {gamma_dis=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb_dis = lamb_dis
            self.gamma_dis = gamma_dis
            return posterior_Qv, posterior_rho
        else:
            raise Exception(f'Warning, optimize_rho: unknown bound {bound}! expected one of {allowed_bounds}')

    def bound(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, alpha=1.0):
        if bound not in ['Uniform', 'Lambda', 'TND_DIS', 'TND', 'DIS']:
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
                
        # for i in range(self.nb_views):
        #     print(f"{labeled_data[0][i].shape=}, {ulX[i].shape=}")
        #     assert  np.array_equal(labeled_data[0][i], ulX[i])
            
        # Compute the Kullback-Leibler divergences
        with torch.no_grad():
            prior_pi = uniform_distribution(v).to(device)
            prior_Pv = [uniform_distribution(m).to(device)]*v
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
            
        if bound == 'Uniform':
            emp_risks_views, emp_mv_risk, ns = self.mv_risks(labeled_data, incl_oob)
            
            # Compute the PB-lambda bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.PBkl_MV(emp_mv_risk, ns, KL_QP.item(), KL_rhopi.item()), emp_mv_risk, -1, KL_QP.item(), KL_rhopi.item(), ns, -1
            else:
                return br.PBkl_MV(emp_mv_risk, ns, RD_QP.item(), RD_rhopi.item()), emp_mv_risk, -1, RD_QP.item(), RD_rhopi.item(), ns, -1
            
        elif bound == 'Lambda':
            emp_risks_views, emp_mv_risk, ns = self.mv_risks(labeled_data, incl_oob)
            
            # Compute the PB-lambda bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.PBkl_MV(emp_mv_risk, ns, KL_QP.item(), KL_rhopi.item()), emp_mv_risk, -1, KL_QP.item(), KL_rhopi.item(), ns, -1
            else:
                return br.PBkl_MV(emp_mv_risk, ns, RD_QP.item(), RD_rhopi.item()), emp_mv_risk, -1, RD_QP.item(), RD_rhopi.item(), ns, -1
            
        elif bound == 'TND_DIS':
            emp_trisks_views, emp_mv_trisk, nt = self.mv_tandem_risk(labeled_data, incl_oob)
            emp_dis_views, emp_mv_dis, nd = self.mv_disagreement(ulX, incl_oob)
            # emp_risks_views, emp_mv_risk, ns = self.mv_risks(labeled_data, incl_oob)
            # print(f"###{emp_mv_risk=} {emp_mv_trisk+0.5*emp_mv_dis=}  {emp_mv_trisk=} {emp_mv_dis=}")
            
            # Compute the TND_DIS bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.TND_DIS_MV(emp_mv_trisk, emp_mv_dis, nt, nd, KL_QP.item(), KL_rhopi.item()), emp_mv_trisk, emp_mv_dis, KL_QP.item(), KL_rhopi.item(), nt, nd
            else:
                return br.TND_DIS_MV(emp_mv_trisk, emp_mv_dis, nt, nd, RD_QP.item(), RD_rhopi.item()), emp_mv_trisk, emp_mv_dis, RD_QP.item(), RD_rhopi.item(), nt, nd
                
        elif bound == 'TND':
            emp_trisks_views, emp_mv_trisk, nt = self.mv_tandem_risk(labeled_data, incl_oob)
            
            # Compute the TND bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.TND_MV(emp_mv_trisk, nt, KL_QP.item(), KL_rhopi.item()), emp_mv_trisk, -1, KL_QP.item(), KL_rhopi.item(), nt, -1
            else:
                return br.TND_MV(emp_mv_trisk, nt, RD_QP.item(), RD_rhopi.item()), emp_mv_trisk, -1, RD_QP.item(), RD_rhopi.item(), nt, -1
            
        elif bound == 'DIS':
            emp_risks_views, emp_mv_risk, ng = self.mv_risks(labeled_data, incl_oob)
            emp_dis_views, emp_mv_dis, nd = self.mv_disagreement(ulX, incl_oob)
            
            # Compute the DIS bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.DIS_MV(emp_mv_risk, emp_mv_dis, ng, nd, KL_QP.item(), KL_rhopi.item()), emp_mv_risk, emp_mv_dis, KL_QP.item(), KL_rhopi.item(), ng, nd
            else:
                return br.DIS_MV(emp_mv_risk, emp_mv_dis, ng, nd, RD_QP.item(), RD_rhopi.item()), emp_mv_risk, emp_mv_dis, RD_QP.item(), RD_rhopi.item(), ng, nd
        
    def set_posteriors(self, posterior_rho, posterior_Qv):
        self.posterior_rho = posterior_rho
        self.posterior_Qv = posterior_Qv
    
    def clear_posteriors(self):
        self.posterior_rho = uniform_distribution(self.nb_views)
        self.posterior_Qv = [uniform_distribution(self.nb_estimators) for _ in range(self.nb_views)]

    def risks(self, data=None, incl_oob=True):
        check_is_fitted(self)
        
        m = self.nb_estimators
        num_views  = self.nb_views
        n = np.zeros((num_views, m))
        risks = np.zeros((num_views, m))
        
        if incl_oob:
            # (preds, targs) = self._OOB
            # # preds = [(idx, preds)] * n_estimators
            # orisk, on = util.oob_risks(preds, targs)
            # n += on
            # risks += orisk
            raise Exception('Warning, risks: OOB not implemented!')

        if data is not None:
            assert (len(data) == 2)
            Xs, Y = data
            P = np.array([[est.predict(X).astype(int) for est in estimators] for X, estimators in zip(Xs, self._estimators_views)])

            risks += util.multiview_risks_(P, Y)
            
            n += Xs[0].shape[0]
            

        return risks, n
    
    def mv_risks(self, labeled_data=None, incl_oob=True):
        risks_views, ns_views = self.risks(labeled_data, incl_oob)
        # print(f"{risks_views=}, {ns_views=}")
        emp_risks_views = np.divide(risks_views, ns_views, where=ns_views!=0)
        # print(f"After {emp_risks_views=}")
        emp_rv = []
        for q, rv in zip(self.posterior_Qv, emp_risks_views):
            emp_rv.append(np.average(rv, weights=q.cpu().detach().numpy(), axis=0))
        
        emp_mv_risk = np.average(emp_rv, weights=self.posterior_rho.cpu().detach().numpy(), axis=0)
        # print(f"Finally {emp_mv_risk=}")
        return np.array(emp_rv), emp_mv_risk, np.min(ns_views)

    
    def mv_tandem_risk(self, labeled_data=None, incl_oob=True):
        trsk, n2 = self.multiview_tandem_risks(labeled_data, incl_oob)
        trsk = np.divide(trsk, n2, where=n2!=0)
        
        emp_tnd_v = np.zeros((self.nb_views, self.nb_views))
        for i in range(self.nb_views):
            qv1 = self.posterior_Qv[i].cpu().detach().numpy()
            for j in range(self.nb_views):
                qv2 = self.posterior_Qv[j].cpu().detach().numpy()
                emp_tnd_v[i, j] = np.average(np.average(trsk[i, j], weights=qv1, axis=0), weights=qv2)
                
        emp_mv_dis_risk = np.average(
            np.average(emp_tnd_v, weights=self.posterior_rho.cpu().detach().numpy(), axis=0),
            weights=self.posterior_rho.cpu().detach().numpy())

        return emp_tnd_v, emp_mv_dis_risk, np.min(n2)
    
    def multiview_tandem_risks(self, data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        num_views = self.nb_views
        n2 = np.zeros((num_views, num_views, m, m))
        tandem_risks = np.zeros((num_views, num_views, m, m))

        if incl_oob:
            raise Exception('Warning, multiview_disagreements: OOB not implemented!')

        if data is not None:
            assert (len(data) == 2)
            Xs, Y = data
            # Iterate over views and compute disagreements for each view
            P = np.array([[est.predict(X).astype(int) for est in estimators] for X, estimators in zip(Xs, self._estimators_views)])
            tandem_risks+= util.multiview_tandem_risks(P,  Y)
            n2 += Xs[0].shape[0]

        return tandem_risks, n2
    
    # Returns the disagreement
    def mv_disagreement(self, unlabeled_data=None, incl_oob=True):
        dis, n2 = self.multiview_disagreements(unlabeled_data, incl_oob)
        dis = np.divide(dis, n2, where=n2!=0)
        
        emp_dis_v = np.zeros((self.nb_views, self.nb_views))
        for i in range(self.nb_views):
            qv1 = self.posterior_Qv[i].cpu().detach().numpy()
            for j in range(self.nb_views):
                qv2 = self.posterior_Qv[j].cpu().detach().numpy()
                emp_dis_v[i, j] = np.average(np.average(dis[i, j], weights=qv1, axis=0), weights=qv2)
                
        emp_mv_dis_risk = np.average(
            np.average(emp_dis_v, weights=self.posterior_rho.cpu().detach().numpy(), axis=0),
            weights=self.posterior_rho.cpu().detach().numpy())

        return emp_dis_v, emp_mv_dis_risk, np.min(n2)
    
    # in the class
    def multiview_disagreements(self, unlabeled_data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        num_views = self.nb_views
        n2 = np.zeros((num_views, num_views, m, m))
        disagreements = np.zeros((num_views, num_views, m, m))

        if incl_oob:
            raise Exception('Warning, multiview_disagreements: OOB not implemented!')

        if unlabeled_data is not None:
            # Iterate over views and compute disagreements for each view
            P = np.array([[est.predict(X).astype(int) for est in estimators] for X, estimators in zip(unlabeled_data, self._estimators_views)])
            disagreements+= util.multiview_disagreements(P)
            n2 += unlabeled_data[0].shape[0]

        return disagreements, n2