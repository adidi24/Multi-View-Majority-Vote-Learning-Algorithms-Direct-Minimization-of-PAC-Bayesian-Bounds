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


    


class MajorityVoteBoundsDeepNeuralDecisionForests(BaseEstimator, ClassifierMixin):
    
    def __init__(self, nb_estimators, depth, used_feature_rate, random_state=42, posterior_Q=None, epochs=100, learning_rate=0.001, use_dndf=False):
        super(MajorityVoteBoundsDeepNeuralDecisionForests, self).__init__()
        self.random_state = random_state
        self._prng = check_random_state(self.random_state)
        self.nb_estimators = nb_estimators
        self._estimators = None
        self.classes_ = None
        self.depth = depth
        self.used_feature_rate = used_feature_rate
        self.posterior_Q = posterior_Q if posterior_Q is not None else uniform_distribution(nb_estimators)
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
        X, y = check_X_y(X, y)
        
        
        # Create estimators for each views
        if self.use_dndf:
            self._estimators = [
                DeepNeuralDecisionForests(depth=self.depth, n_in_feature=X.shape[1], used_feature_rate=self.used_feature_rate, epochs=self.epochs, learning_rate=self.learning_rate)
                for _ in range(self.nb_estimators)]
        else:
            self._estimators = [
                Tree(rand=self._prng, max_features="sqrt", max_depth=self.depth)
                for _ in range(self.nb_estimators)]
        
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        preds = []
        n = self.X_.shape[0]  # Number of samples
        
        for est in self._estimators:
            # Sample points for training (w. replacement)
            while True:
                t_idx = self._prng.randint(n, size=n)
                t_X = self.X_[t_idx]
                t_Y = self.y_[t_idx]
                if np.unique(t_Y).shape[0] == len(self.classes_):
                    break

            oob_idx = np.delete(np.arange(n), np.unique(t_idx))
            oob_X = self.X_[oob_idx]
            
            est = est.fit(t_X, t_Y)  # Fit this estimator
            oob_P = est.predict(oob_X)  # Predict on OOB

            M_est = np.zeros(self.y_.shape)
            P_est = np.zeros(self.y_.shape)
            M_est[oob_idx] = 1
            P_est[oob_idx] = oob_P
            preds.append((M_est, P_est))
            
        # print(f'View {i+1}/{self.nb} done!')
        self._OOB = (preds, self.y_)
        return self

    def predict(self, Xs, Y=None):
        """
        Return the predicted class labels using majority vote of the
        predictions.
        """
        check_is_fitted(self)
        
        Q = self.posterior_Q.cpu().data.numpy()
        
        if Y is not None:
            Xs, Y = check_X_y(Xs, Y)
        else:
            Xs = check_array(Xs)

        P = [est.predict(Xs).astype(int) for est in self._estimators]
        mvtP = util.mv_preds(Q, np.array(P))

        # print(f"Xs shapes: {[x.shape for x in Xs]=}\n\n {Y.shape=}\n\n {[y.shape for y in ys]=}\n\n {len(ys)=}\n\n {len(mvP)=}")
        return (mvtP, util.risk(mvtP, Y)) if Y is not None else mvtP
    
    def  optimize_Q(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, max_iter=1000, optimise_lambda_gamma=False, alpha=1):
        allowed_bounds = {'Lambda', 'TND_DIS', 'TND', 'DIS'}
        if bound not in allowed_bounds:
            raise Exception(f'Warning, optimize_Q: unknown bound {bound}! expected one of {allowed_bounds}')
        if labeled_data is None and not incl_oob:
            raise Exception('Warning, stats: Missing data! expected labeled_data or incl_oob=True')

        check_is_fitted(self)
        
        ulX =  None
        if unlabeled_data is not None:
            if labeled_data is None:
                ulX = unlabeled_data  
            else:
                ulX = np.concatenate((labeled_data[0], unlabeled_data), axis=0)
        else:
            if labeled_data is not None:
                ulX = labeled_data[0]
            
        
        if bound == 'Lambda':
            risks, ns = self.risks(labeled_data, incl_oob)
            emp_risks = np.divide(risks, ns, where=ns!=0)
            ns_min = torch.tensor(np.min(ns))

            if alpha == 1:
                posterior_Q, lamb = bkl.optimizeLamb_torch(emp_risks, ns_min, device, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma)
            else:
                posterior_Q, lamb = br.optimizeLamb_torch(emp_risks, ns_min, device, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb=}")
            self.set_posteriors(posterior_Q)
            self.lamb = lamb
            return posterior_Q
        
        elif bound == 'TND_DIS':
            trisks, ns_t = self.tandem_risks(labeled_data, incl_oob)
            dis, ns_d = self.disagreements(ulX, incl_oob)
            emp_trisks = np.divide(trisks, ns_t, where=ns_t!=0)
            emp_dis = np.divide(dis, ns_d, where=ns_d!=0)
            nt = torch.tensor(np.min(ns_t))
            nd = torch.tensor(np.min(ns_d))

            if alpha == 1:
                posterior_Q, lamb1_tnd_dis, lamb2_tnd_dis = bkl.optimizeTND_DIS_torch(emp_trisks, emp_dis, nt, nd, device, max_iter=max_iter, optimise_lambdas=optimise_lambda_gamma)
            else:
                posterior_Q, lamb1_tnd_dis, lamb2_tnd_dis = br.optimizeTND_DIS_torch(emp_trisks, emp_dis, nt, nd, device, max_iter=max_iter, optimise_lambdas=optimise_lambda_gamma, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            # print(f"{lamb1_tnd_dis=}, {lamb2_tnd_dis=}")
            self.lamb1_tnd_dis = lamb1_tnd_dis
            self.lamb2_tnd_dis = lamb2_tnd_dis
            return posterior_Q
        
        elif bound == 'TND':
            trisks, ns = self.tandem_risks(labeled_data, incl_oob)
            emp_trisks = np.divide(trisks, ns, where=ns!=0)
            ns_min = torch.tensor(np.min(ns))

            if alpha == 1:
                posterior_Q, lamb_tnd = bkl.optimizeTND_torch(emp_trisks, ns_min, device, max_iter=max_iter, optimise_lambda=optimise_lambda_gamma)
            else:
                posterior_Q, lamb_tnd = br.optimizeTND_torch(emp_trisks, ns_min, device, max_iter=max_iter, optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_tnd=}")
            self.set_posteriors(posterior_Q)
            self.lamb_tnd = lamb_tnd
            return posterior_Q
        
        elif bound == 'DIS':
            risks, ns_g = self.risks(labeled_data, incl_oob)
            dis, ns_d = self.disagreements(ulX, incl_oob)
            emp_risks = np.divide(risks, ns_g, where=ns_g!=0)
            emp_dis = np.divide(dis, ns_d, where=ns_d!=0)
            ng = torch.tensor(np.min(ns_g))
            nd = torch.tensor(np.min(ns_d))

            if alpha == 1:
                posterior_Q, lamb_dis, gamma_dis = bkl.optimizeDIS_torch(emp_risks, emp_dis, ng, nd, device, max_iter=max_iter, optimise_lambda_gamma=optimise_lambda_gamma)
            else:
                posterior_Q, lamb_dis, gamma_dis = br.optimizeDIS_torch(emp_risks, emp_dis, ng, nd, device, max_iter=max_iter, optimise_lambda_gamma=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_dis=}, {gamma_dis=}")
            self.set_posteriors(posterior_Q)
            self.lamb_dis = lamb_dis
            self.gamma_dis = gamma_dis
            return posterior_Q
        else:
            raise Exception(f'Warning, optimize_Q: unknown bound {bound}! expected one of {allowed_bounds}')

    def bound(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, alpha=1.0):
        if bound not in ['Uniform', 'Lambda', 'TND_DIS', 'TND', 'DIS']:
            raise Exception("Warning, ViewClassifier.bound: Unknown bound!")
        
        m = self.nb_estimators
        
        ulX =  None
        if unlabeled_data is not None:
            if labeled_data is None:
                ulX = unlabeled_data  
            else:
                ulX = np.concatenate((labeled_data[0], unlabeled_data), axis=0)
        else:
            if labeled_data is not None:
                ulX = labeled_data[0]
                
        
        # Compute the Kullback-Leibler divergences
        with torch.no_grad():
            prior_P = uniform_distribution(m).to(device)
            if alpha==1:
                KL_QP = kl(self.posterior_Q, prior_P).item()
            else:
                RD_QP = rd(self.posterior_Q, prior_P, alpha).item()
        
        # print(f"{KL_QP=},  {KL_QP=}")
        if bound == 'Uniform':
            emp_risk, n_min = self.gibbs_risk(labeled_data, incl_oob)
            
            # Compute the PB-lambda bound for each view and for the multiview resp.
            if alpha==1:
                return (bkl.PBkl(emp_risk, n_min, KL_QP)), emp_risk, -1, KL_QP, n_min, -1
            else:
                return (br.PBkl(emp_risk, n_min, RD_QP)), emp_risk, -1, RD_QP, n_min, -1
            
        elif bound == 'Lambda':
            emp_risk, n_min = self.gibbs_risk(labeled_data, incl_oob)
            
            # Compute the PB-lambda bound for each view and for the multiview resp.
            if alpha==1:
                return (bkl.PBkl(emp_risk, n_min, KL_QP)), emp_risk, -1, KL_QP, n_min, -1
            else:
                return (br.PBkl(emp_risk, n_min, RD_QP)), emp_risk, -1, RD_QP, n_min, -1
            
        elif bound == 'TND_DIS':
            emp_trisk, nt = self.tandem_risk(labeled_data, incl_oob)
            emp_dis, nd = self.disagreement(ulX, incl_oob)
            # emp_risk, n_min = self.gibbs_risk(labeled_data, incl_oob)
            # print(f"{emp_risk=}, {emp_trisk+0.5*emp_dis=}, {emp_trisk=}, {emp_dis=}")
            
            # Compute the TND_DIS bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.TND_DIS(emp_trisk, emp_dis, nt, nd, KL_QP), emp_trisk, emp_dis, KL_QP, nt, nd
            else:
                return br.TND_DIS(emp_trisk, emp_dis, nt, nd, RD_QP), emp_trisk, emp_dis, RD_QP, nt, nd
        elif bound == 'TND':
            emp_trisk, n_min = self.tandem_risk(labeled_data, incl_oob)
            
            # Compute the TND bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.TND(emp_trisk, n_min, KL_QP), emp_trisk, -1, KL_QP, n_min, -1
            else:
                return br.TND(emp_trisk, n_min, RD_QP), emp_trisk, -1, RD_QP, n_min, -1
        elif bound == 'DIS':
            emp_risk, ng = self.gibbs_risk(labeled_data, incl_oob)
            emp_dis, nd = self.disagreement(ulX, incl_oob)
            
            # Compute the DIS bound for each view and for the multiview resp.
            if alpha==1:
                return bkl.DIS(emp_risk, emp_dis, ng, nd, KL_QP), emp_risk, emp_dis, KL_QP, ng, nd
            else:
                return br.DIS(emp_risk, emp_dis, ng, nd, RD_QP), emp_risk, emp_dis, RD_QP, ng, nd
        
    def set_posteriors(self, posterior_Q):
        self.posterior_Q = posterior_Q
    
    def clear_posteriors(self):
        self.posterior_Q = uniform_distribution(self.nb_estimators)


    def gibbs_risk(self, labeled_data=None, incl_oob=True):
        risks, ns = self.risks(labeled_data, incl_oob)
        posterior_Q = self.posterior_Q.cpu().detach().numpy()
        emp_risk = np.average(risks/ns, weights=posterior_Q, axis=0)
        return emp_risk, np.min(ns)
    
    def risks(self, data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        n = np.zeros((m,))
        risks = np.zeros((m,))
        if incl_oob:
            (preds, targs) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            orisk, on = util.oob_risks(preds, targs)
            n += on
            risks += orisk

        if data is not None:
            assert (len(data) == 2)
            X, Y = data
            P = np.array([est.predict(X).astype(int) for est in self._estimators])

            n += X.shape[0]
            risks += util.risks_(P, Y)

        return risks, n

    
    def tandem_risk(self, labeled_data=None, incl_oob=True):
        trsk, nt = self.tandem_risks(labeled_data, incl_oob)
        
        posterior_Q = self.posterior_Q.cpu().detach().numpy()
        emp_tnd_risk = np.average(
            np.average(trsk/nt, weights=posterior_Q, axis=0),
            weights=posterior_Q)

        return emp_tnd_risk, np.min(nt)

    def tandem_risks(self, data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        n2 = np.zeros((m, m))
        tandem_risks = np.zeros((m, m))

        if incl_oob:
            (preds, targs) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            otand, on2 = util.oob_tandem_risks(preds, targs)
            n2 += on2
            tandem_risks += otand

        if data is not None:
            assert (len(data) == 2)
            X, Y = data
            P = np.array([est.predict(X).astype(int) for est in self._estimators])

            n2 += X.shape[0]
            tandem_risks += util.tandem_risks(P, Y)

        return tandem_risks, n2
    
    # Returns the disagreement
    def disagreement(self, unlabeled_data=None, incl_oob=True):
        dis, nd = self.disagreements(unlabeled_data, incl_oob)
        
        posterior_Q = self.posterior_Q.cpu().detach().numpy()
        emp_dis = np.average(
            np.average(dis/nd, weights=posterior_Q, axis=0), weights=posterior_Q)

        return emp_dis, np.min(nd)

    def disagreements(self, unlabeled_data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        n2 = np.zeros((m, m))
        disagreements = np.zeros((m, m))

        if incl_oob:
            (preds, Y) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            odis, on2 = util.oob_disagreements(preds)
            n2 += on2
            disagreements += odis

        if unlabeled_data is not None:
            X = unlabeled_data
            P = np.array([est.predict(X).astype(int) for est in self._estimators])

            n2 += X.shape[0]
            disagreements += util.disagreements(P)

        return disagreements, n2