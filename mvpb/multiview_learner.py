import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import check_random_state

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# import mvpb.bounds_alpha1_KL as bkl
import mvpb.bounds as bounds
import mvpb.util as util
from .bounds.tools import renyi_divergence as rd, kl
from .util import uniform_distribution

from .deepTree import DeepNeuralDecisionForests
from .tree import Tree
    


class MultiViewMajorityVoteLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, nb_estimators, nb_views, depth, used_feature_rate, random_state=42, posterior_rho=None, epochs=100, learning_rate=0.001, use_dndf=False):
        super(MultiViewMajorityVoteLearner, self).__init__()
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
        self.lamb_eS = None
        self.lamb1_eS_dS = None
        self.lamb2_eS_dS = None
        self.lamb_dS = None
        self.gamma_dS = None
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

    def predict_MV(self, Xs, Y=None):
        """
        Return the predicted class labels using majority vote of the
        predictions from each view.
        """
        check_is_fitted(self)
        
        rho = self.posterior_rho.cpu().data.numpy()
        posteriors_qs = [p.cpu().data.numpy() for p in self.posterior_Qv]
        
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

        # print(f"Xs shapes: {[x.shape for x in Xs]=}\n\n {Y.shape=}\n\n {[y.shape for y in ys]=}\n\n {len(ys)=}\n\n {len(mvP)=}")
        return (mvP, util.risk(mvP, Y)) if Y is not None else mvP
    
    def optimize_rho(self, 
                     bound,
                     labeled_data=None,
                     unlabeled_data=None,
                     incl_oob=True,
                     max_iter=1000,
                     optimise_lambda_gamma=False,
                     alpha=1,
                     t=100):
        """
        Optimize the value of rho (hyper-posterior distribution) and Q (posterior distribution)
        based on the specified bound.

        Parameters:
        - bound (str): The bound to optimize rho for.
            Must be one of {'PBkl', 'PBkl_inv', 'TND_DIS', 'TND_DIS_inv', 'TND', 'TND_inv', 'DIS', 'DIS_inv'}.
        - labeled_data (tuple or None): The labeled data to use for optimization. Default is None.
        - unlabeled_data (list or None): The unlabeled data to use for optimization (in the disagreement case). Default is None.
        - incl_oob (bool): Whether to include out-of-bag samples in the optimization. Default is True.
        - max_iter (int): The maximum number of iterations for the optimization algorithm. Default is 1000.
        - optimise_lambda_gamma (bool): Whether to optimize lambda and gamma parameters. Default is False.
        - alpha (float): The alpha parameter for the optimization algorithm (1 to use KL divergence, otherwise Renyi divergence). Default is 1.

        Returns:
        - posterior_Qv (list): The optimized posterior distributions for each view.
        - posterior_rho (float): The optimized hyper-posterior rho distribution.
        """
        
        allowed_bounds = {'PBkl', 'PBkl_inv', 'TND_DIS', 'TND_DIS_inv', 'TND', 'TND_inv', 'DIS', 'DIS_inv', 'Cbound', 'C_TND'}
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
                
        
        risks_views, ns_views_g = self.risks(labeled_data, incl_oob)
        dis_views, ns_views_d = self.multiview_disagreements(ulX, incl_oob)
        multiview_joint_errors, ns_views_t = self.multiview_joint_errors(labeled_data, incl_oob)
        
        grisks_views = np.divide(risks_views, ns_views_g, where=ns_views_g!=0)
        dS_views = np.divide(dis_views, ns_views_d, where=ns_views_d!=0)
        eS_views = np.divide(multiview_joint_errors, ns_views_t, where=ns_views_t!=0)
        
        ng = torch.tensor(np.min(ns_views_g))
        nd = torch.tensor(np.min(ns_views_d))
        ne = torch.tensor(np.min(ns_views_t))

        
        if bound == 'PBkl':
            posterior_Qv, posterior_rho, lamb = bounds.fo.optimizeLamb_mv_torch(grisks_views, ng, device, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma, alpha=alpha, t=t)
            
            # print(f"{lamb=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb = lamb
            return posterior_Qv, posterior_rho

        elif bound == 'PBkl_inv':
            posterior_Qv, posterior_rho = bounds.fo.optimizeKLinv_mv_torch(grisks_views, ng, device, max_iter=max_iter, alpha=alpha, t=t)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND_DIS':
            posterior_Qv, posterior_rho, lamb1_eS_dS, lamb2_eS_dS = bounds.fo.optimizeTND_DIS_mv_torch(eS_views, dS_views, ne, nd, device, max_iter=max_iter, optimise_lambdas=optimise_lambda_gamma, alpha=alpha, t=t)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb1_eS_dS = lamb1_eS_dS
            self.lamb2_eS_dS = lamb2_eS_dS
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND_DIS_inv':
            posterior_Qv, posterior_rho = bounds.fo.optimizeTND_DIS_inv_mv_torch(eS_views, dS_views, ne, nd, device, max_iter=max_iter, alpha=alpha, t=t)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        elif bound == 'TND':
            posterior_Qv, posterior_rho, lamb_eS = bounds.so.optimizeTND_mv_torch(eS_views, ne, device, max_iter=max_iter, optimise_lambda=optimise_lambda_gamma, alpha=alpha, t=t)
            
            # print(f"{lamb_eS=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb_eS = lamb_eS
            return posterior_Qv, posterior_rho

        elif bound == 'TND_inv':
            posterior_Qv, posterior_rho= bounds.so.optimizeTND_Inv_mv_torch(eS_views, ne, device, max_iter=max_iter, alpha=alpha, t=t)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        elif bound == 'DIS':
            posterior_Qv, posterior_rho, lamb_dS, gamma_dS = bounds.so.optimizeDIS_mv_torch(grisks_views, dS_views, ng, nd, device, max_iter=max_iter, optimise_lambda_gamma=optimise_lambda_gamma, alpha=alpha, t=t)
            
            # print(f"{lamb_dS=}, {gamma_dS=}")
            self.set_posteriors(posterior_rho, posterior_Qv)
            self.lamb_dS = lamb_dS
            self.gamma_dS = gamma_dS
            return posterior_Qv, posterior_rho
        
        elif bound == 'DIS_inv':
            posterior_Qv, posterior_rho = bounds.so.optimizeDIS_Inv_mv_torch(grisks_views, dS_views, ng, nd, device, max_iter=max_iter, alpha=alpha, t=t)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        elif bound == 'Cbound':
            posterior_Qv, posterior_rho = bounds.cb.optimizeCBound_mv_torch(grisks_views, dS_views, ng, nd, device, max_iter=max_iter, alpha=alpha, t=1)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        elif bound == 'C_TND':
            posterior_Qv, posterior_rho = bounds.cb.optimizeCTND_mv_torch(grisks_views, eS_views, ng, ne, device, max_iter=max_iter, alpha=alpha, t=t)
            
            self.set_posteriors(posterior_rho, posterior_Qv)
            return posterior_Qv, posterior_rho
        
        else:
            raise Exception(f'Warning, optimize_rho: unknown bound {bound}! expected one of {allowed_bounds}')

    def bound(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, alpha=1.0):
        allowed_bounds = {'Uniform', 'PBkl', 'PBkl_inv', 'TND_DIS', 'TND_DIS_inv', 'TND', 'TND_inv', 'DIS', 'DIS_inv', 'Cbound', 'C_TND'}
        if bound not in allowed_bounds:
            raise Exception(f'Warning, bound: unknown bound {bound}! expected one of {allowed_bounds}')
        
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
            prior_pi = uniform_distribution(v).to(device)
            prior_Pv = [uniform_distribution(m).to(device)]*v
            if alpha==1:
                DIV_QPs = [kl(q, p)  for q, p in zip(self.posterior_Qv, prior_Pv)]
                DIV_QP = torch.sum(torch.stack(DIV_QPs) * self.posterior_rho)
                # print(f"{self.posterior_rho=},  {prior_pi=}")
                DIV_rhopi = kl(self.posterior_rho, prior_pi)
            else:
                DIV_QPs = [rd(q, p, alpha)  for q, p in zip(self.posterior_Qv, prior_Pv)]
                DIV_QP = torch.sum(torch.stack(DIV_QPs) * self.posterior_rho)
                DIV_rhopi = rd(self.posterior_rho, prior_pi, alpha)
        # print(f"{DIV_rhopi=},  {DIV_QP=}")
            
        _, emp_mv_risk, ng = self.mv_risk(labeled_data, incl_oob)
        _, eS, ne = self.mv_expected_joint_error(labeled_data, incl_oob)
        _, dS, nd = self.mv_expected_disagreement(ulX, incl_oob)

        stats = (emp_mv_risk, eS, dS, DIV_QP.item(), DIV_rhopi.item(), ng, ne, nd,)

        if bound == 'Uniform':
            # Compute the MV PB-lambda bound.
            return (bounds.fo.PBkl_MV(emp_mv_risk, ng, DIV_QP.item(), DIV_rhopi.item()),) + stats
            
        if bound == 'PBkl':
            # Compute the MV PB-lambda bound.
            return (bounds.fo.PBkl_MV(emp_mv_risk, ng, DIV_QP.item(), DIV_rhopi.item()),) + stats
            
        elif bound == 'PBkl_inv':
            # Compute the MV PB-lambda bound.
            return (bounds.fo.KLInv_MV(emp_mv_risk, ng, DIV_QP.item(), DIV_rhopi.item()),) + stats
            
        elif bound == 'TND_DIS':
            # Compute the MV TND_DIS bound.
            return (bounds.fo.TND_DIS_MV(eS, dS, ne, nd, DIV_QP.item(), DIV_rhopi.item()),) + stats
            
        elif bound == 'TND_DIS_inv':
            # Compute the MV TND_DIS bound.
            return (bounds.fo.TND_DIS_Inv_MV(eS, dS, ne, nd, DIV_QP.item(), DIV_rhopi.item()),) + stats
        
        elif bound == 'TND':
            # Compute the MV TND bound.
            return (bounds.so.TND_MV(eS, ne, DIV_QP.item(), DIV_rhopi.item()),) + stats

        elif bound == 'TND_inv':
            # Compute the MV TND Inv bound.
            return (bounds.so.TND_Inv_MV(eS, ne, DIV_QP.item(), DIV_rhopi.item()),) + stats
            
        elif bound == 'DIS':
            # Compute the MV DIS bound for
            return (bounds.so.DIS_MV(emp_mv_risk, dS, ng, nd, DIV_QP.item(), DIV_rhopi.item()),) + stats
        
        elif bound == 'DIS_inv':
            # Compute the MV DIS bound.
            return (bounds.so.DIS_Inv_MV(emp_mv_risk, dS, ng, nd, DIV_QP.item(), DIV_rhopi.item()),) + stats

        elif bound == 'Cbound':
            # Compute the MV C-bound.
            return (bounds.cb.Cbound_MV(emp_mv_risk, dS, ng, nd, DIV_QP.item(), DIV_rhopi.item()),) + stats
        
        elif bound == 'C_TND':
            # Compute the MV C-tandem bound.
            return (bounds.cb.C_TND_MV(emp_mv_risk, eS, ng, ne, DIV_QP.item(), DIV_rhopi.item()),) + stats
        
    def set_posteriors(self, posterior_rho, posterior_Qv):
        self.posterior_rho = posterior_rho
        self.posterior_Qv = posterior_Qv
    
    def clear_posteriors(self):
        self.posterior_rho = uniform_distribution(self.nb_views).to(device)
        self.posterior_Qv = [uniform_distribution(self.nb_estimators).to(device) for _ in range(self.nb_views)]

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
    
    def mv_risk(self, labeled_data=None, incl_oob=True):
        risks_views, ns_views = self.risks(labeled_data, incl_oob)
        # print(f"{risks_views=}, {ns_views=}")
        grisks_views = np.divide(risks_views, ns_views, where=ns_views!=0)
        # print(f"After {grisks_views=}")
        emp_rv = []
        for q, rv in zip(self.posterior_Qv, grisks_views):
            emp_rv.append(np.average(rv, weights=q.cpu().detach().numpy(), axis=0))
        
        emp_mv_risk = np.average(emp_rv, weights=self.posterior_rho.cpu().detach().numpy(), axis=0)
        # print(f"Finally {emp_mv_risk=}")
        return np.array(emp_rv), emp_mv_risk, np.min(ns_views)

    
    def mv_expected_joint_error(self, labeled_data=None, incl_oob=True):
        multiview_joint_errors, n2 = self.multiview_joint_errors(labeled_data, incl_oob)
        multiview_joint_errors = np.divide(multiview_joint_errors, n2, where=n2!=0)
        
        eS_v = np.zeros((self.nb_views, self.nb_views))
        for i in range(self.nb_views):
            qv1 = self.posterior_Qv[i].cpu().detach().numpy()
            for j in range(self.nb_views):
                qv2 = self.posterior_Qv[j].cpu().detach().numpy()
                eS_v[i, j] = np.average(np.average(multiview_joint_errors[i, j], weights=qv1, axis=0), weights=qv2)
                
        eS = np.average(
            np.average(eS_v, weights=self.posterior_rho.cpu().detach().numpy(), axis=0),
            weights=self.posterior_rho.cpu().detach().numpy())

        return eS_v, eS, np.min(n2)
    
    def multiview_joint_errors(self, data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        num_views = self.nb_views
        n2 = np.zeros((num_views, num_views, m, m))
        multiview_joint_errors = np.zeros((num_views, num_views, m, m))

        if incl_oob:
            raise Exception('Warning, multiview_joint_errors: OOB not implemented!')

        if data is not None:
            assert (len(data) == 2)
            Xs, Y = data
            # Iterate over views and compute disagreements for each view
            P = np.array([[est.predict(X).astype(int) for est in estimators] for X, estimators in zip(Xs, self._estimators_views)])
            multiview_joint_errors+= util.multiview_joint_errors(P,  Y)
            n2 += Xs[0].shape[0]

        return multiview_joint_errors, n2
    
    # Returns the disagreement
    def mv_expected_disagreement(self, unlabeled_data=None, incl_oob=True):
        multiview_disagreements, n2 = self.multiview_disagreements(unlabeled_data, incl_oob)
        multiview_disagreements = np.divide(multiview_disagreements, n2, where=n2!=0)
        
        dS_v = np.zeros((self.nb_views, self.nb_views))
        for i in range(self.nb_views):
            qv1 = self.posterior_Qv[i].cpu().detach().numpy()
            for j in range(self.nb_views):
                qv2 = self.posterior_Qv[j].cpu().detach().numpy()
                dS_v[i, j] = np.average(np.average(multiview_disagreements[i, j], weights=qv1, axis=0), weights=qv2)
                
        dS = np.average(
            np.average(dS_v, weights=self.posterior_rho.cpu().detach().numpy(), axis=0),
            weights=self.posterior_rho.cpu().detach().numpy())

        return dS_v, dS, np.min(n2)
    
    # in the class
    def multiview_disagreements(self, unlabeled_data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        num_views = self.nb_views
        n2 = np.zeros((num_views, num_views, m, m))
        multiview_disagreements = np.zeros((num_views, num_views, m, m))

        if incl_oob:
            raise Exception('Warning, multiview_disagreements: OOB not implemented!')

        if unlabeled_data is not None:
            # Iterate over views and compute disagreements for each view
            P = np.array([[est.predict(X).astype(int) for est in estimators] for X, estimators in zip(unlabeled_data, self._estimators_views)])
            multiview_disagreements+= util.multiview_disagreements(P)
            n2 += unlabeled_data[0].shape[0]

        return multiview_disagreements, n2