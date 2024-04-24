import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin



###############################################################################


class Tree(BaseEstimator, ClassifierMixin):

    def __init__(self, rand=None, max_features="sqrt", max_depth=None):
        # Same tree as [1]
        # (see https://github.com/StephanLorenzen/MajorityVoteBounds/
        # blob/master/mvb/rfc.py)
        self.tree = DecisionTreeClassifier(
            criterion="gini",
            max_features=max_features,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=max_depth,
            random_state=rand)

    def fit(self, X, y):
        """
        Run the algorithm CB-Boost

        Parameters
        ----------
        X: ndarray
            The inputs
        y: ndarray
            The labels
        """
        self.tree.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self.tree)
        X = check_array(X)
        
        pred = self.tree.predict(X)
        
        return pred