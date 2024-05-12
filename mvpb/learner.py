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



class MajorityVoteLearner(BaseEstimator, ClassifierMixin):
    
    def __init__(self, nb_estimators, depth, used_feature_rate, random_state=42, posterior_Q=None, epochs=100, learning_rate=0.001, use_dndf=False):
        super(MajorityVoteLearner, self).__init__()
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
        self.lamb_eS = None
        self.lamb1_eS_dS = None
        self.lamb2_eS_dS = None
        self.lamb_dS = None
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
        allowed_bounds = {'PBkl', 'PBkl_inv', 'TND_DIS', 'TND_DIS_inv', 'TND', 'TND_inv', 'DIS', 'DIS_inv', 'Cbound', 'C_TND'}
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
            
        risks, ng = self.risks(labeled_data, incl_oob)
        disagreements, nd = self.disagreements(ulX, incl_oob)
        joint_errors, ne = self.joint_errors(labeled_data, incl_oob)
        
        emp_risks = np.divide(risks, ng, where=ng!=0)
        emp_disagreements = np.divide(disagreements, nd, where=nd!=0)
        emp_joint_errors = np.divide(joint_errors, ne, where=ne!=0)
        
        ng = torch.tensor(np.min(ng))
        nd = torch.tensor(np.min(nd))
        ne = torch.tensor(np.min(ne))
            
        
        if bound == 'PBkl':

            posterior_Q, lamb = bounds.fo.optimizeLamb_torch(emp_risks, ng, device, max_iter=max_iter,  optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb=}")
            self.set_posteriors(posterior_Q)
            self.lamb = lamb
            return posterior_Q

        elif bound == 'PBkl_inv':
            posterior_Q = bounds.fo.optimizeKLinv_torch(emp_risks, ng, device, max_iter=max_iter, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            return posterior_Q

        elif bound == 'TND_DIS':
            posterior_Q, lamb1_eS_dS, lamb2_eS_dS = bounds.fo.optimizeTND_DIS_torch(emp_joint_errors, emp_disagreements, ne, nd, device, max_iter=max_iter, optimise_lambdas=optimise_lambda_gamma, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            self.lamb1_eS_dS = lamb1_eS_dS
            self.lamb2_eS_dS = lamb2_eS_dS
            return posterior_Q

        elif bound == 'TND_DIS_inv':
            posterior_Q = bounds.fo.optimizeTND_DIS_inv_torch(emp_joint_errors, emp_disagreements, ne, nd, device, max_iter=max_iter, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            return posterior_Q
        
        elif bound == 'TND':
            posterior_Q, lamb_eS = bounds.so.optimizeTND_torch(emp_joint_errors, ne, device, max_iter=max_iter, optimise_lambda=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_eS=}")
            self.set_posteriors(posterior_Q)
            self.lamb_eS = lamb_eS
            return posterior_Q

        elif bound == 'TND_inv':
            posterior_Q = bounds.so.optimizeTND_Inv_torch(emp_joint_errors, ne, device, max_iter=max_iter, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            return posterior_Q
        
        elif bound == 'DIS':
            posterior_Q, lamb_dS, gamma_dis = bounds.so.optimizeDIS_torch(emp_risks, emp_disagreements, ng, nd, device, max_iter=max_iter, optimise_lambda_gamma=optimise_lambda_gamma, alpha=alpha)
            
            # print(f"{lamb_dS=}, {gamma_dis=}")
            self.set_posteriors(posterior_Q)
            self.lamb_dS = lamb_dS
            self.gamma_dis = gamma_dis
            return posterior_Q
        
        elif bound == 'DIS_inv':
            posterior_Q = bounds.so.optimizeDIS_Inv_torch(emp_risks, emp_disagreements, ng, nd, device, max_iter=max_iter, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            return posterior_Q

        elif bound == 'Cbound':
            posterior_Q = bounds.cb.optimizeCBound_torch(emp_risks, emp_disagreements, ng, nd, device, max_iter=max_iter, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            return posterior_Q

        elif bound == 'C_TND':
            posterior_Q = bounds.cb.optimizeCTND_torch(emp_risks, emp_joint_errors, ng, ne, device, max_iter=max_iter, alpha=alpha)
            
            self.set_posteriors(posterior_Q)
            return posterior_Q
        
        else:
            raise Exception(f'Warning, optimize_Q: unknown bound {bound}! expected one of {allowed_bounds}')

    def bound(self, bound, labeled_data=None, unlabeled_data=None, incl_oob=True, alpha=1.0):
        allowed_bounds = {'Uniform', 'PBkl', 'PBkl_inv', 'TND_DIS', 'TND_DIS_inv', 'TND', 'TND_inv', 'DIS', 'DIS_inv', 'Cbound', 'C_TND'}
        if bound not in allowed_bounds:
            raise Exception(f'Warning, bound: unknown bound {bound}! expected one of {allowed_bounds}')
        
        
        ulX =  None
        if unlabeled_data is not None:
            if labeled_data is None:
                ulX = unlabeled_data  
            else:
                ulX = np.concatenate((labeled_data[0], unlabeled_data), axis=0)
        else:
            if labeled_data is not None:
                ulX = labeled_data[0]
                
        
        m = self.nb_estimators
        # Compute the Kullback-Leibler divergences
        with torch.no_grad():
            prior_P = uniform_distribution(m).to(device)
            if alpha==1:
                DIV_QP = kl(self.posterior_Q, prior_P).item()
            else:
                DIV_QP = rd(self.posterior_Q, prior_P, alpha).item()
        
        emp_grisk, ng = self.gibbs_risk(labeled_data, incl_oob)
        eS, ne = self.joint_error(labeled_data, incl_oob)
        dS, nd = self.disagreement(ulX, incl_oob)

        stats = (emp_grisk, eS, dS, DIV_QP, ng, ne, nd,)
        
        # print(f"{DIS_QP=},  {DIS_QP=}")
        if bound == 'Uniform':
            return (bounds.fo.PBkl(emp_grisk, ng, DIV_QP),) + stats
            
        if bound == 'PBkl':
            # Compute the PB-kl bound for each view.
            return (bounds.fo.PBkl(emp_grisk, ng, DIV_QP),) + stats

        elif bound == 'PBkl_inv':
            # Compute the PB-kl Invbound for each view.
            return (bounds.fo.KLInv(emp_grisk, ng, DIV_QP),) + stats

        elif bound == 'TND_DIS':
            # Compute the TND_DIS bound for each view.
            return (bounds.fo.TND_DIS(eS, dS, ne, nd, DIV_QP),) + stats

        elif bound == 'TND_DIS_inv':
            # Compute the TND_DIS Inv bound for each view.
            return (bounds.fo.TND_DIS_Inv(eS, dS, ne, nd, DIV_QP),) + stats

        elif bound == 'TND':
            # Compute the TND bound for each view.
            return (bounds.so.TND(eS, ne, DIV_QP),) + stats

        elif bound == 'TND_inv':
            # Compute the TND Inv bound for each view.
            return (bounds.so.TND_Inv(eS, ne, DIV_QP),) + stats
            
        elif bound == 'DIS':
            # Compute the DIS bound for each view.
            return (bounds.so.DIS(emp_grisk, dS, ng, nd, DIV_QP),) + stats
        
        elif bound == 'DIS_inv':
            # Compute the DIS Inv bound for each view.
            return (bounds.so.DIS_Inv(emp_grisk, dS, ng, nd, DIV_QP),) + stats
        
        elif bound == 'Cbound':
            # Compute the C-bound bound for each view.
            return (bounds.cb.Cbound(emp_grisk, dS, ng, nd, DIV_QP),) + stats
        
        elif bound == 'C_TND':
            # Compute the C-tandem bound bound for each view.
            
            return (bounds.cb.C_TND(emp_grisk, eS, ng, ne, DIV_QP),) + stats
        
    def set_posteriors(self, posterior_Q):
        self.posterior_Q = posterior_Q
    
    def clear_posteriors(self):
        self.posterior_Q = uniform_distribution(self.nb_estimators).to(device)


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

    
    def joint_error(self, labeled_data=None, incl_oob=True):
        joint_errors, nt = self.joint_errors(labeled_data, incl_oob)
        
        posterior_Q = self.posterior_Q.cpu().detach().numpy()
        eS = np.average(
            np.average(joint_errors/nt, weights=posterior_Q, axis=0),
            weights=posterior_Q)

        return eS, np.min(nt)

    def joint_errors(self, data=None, incl_oob=True):
        check_is_fitted(self)
        m = self.nb_estimators
        n2 = np.zeros((m, m))
        joint_errors = np.zeros((m, m))

        if incl_oob:
            (preds, targs) = self._OOB
            # preds = [(idx, preds)] * n_estimators
            o_joint, on2 = util.oob_joint_errors(preds, targs)
            n2 += on2
            joint_errors += o_joint

        if data is not None:
            assert (len(data) == 2)
            X, Y = data
            P = np.array([est.predict(X).astype(int) for est in self._estimators])

            n2 += X.shape[0]
            joint_errors += util.joint_errors(P, Y)

        return joint_errors, n2
    
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