import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tqdm.notebook import trange

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin

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
    

class DeepTree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class):
        super(DeepTree, self).__init__()
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
        self.model = DeepTree(self.depth, self.n_in_feature, self.used_feature_rate, self.n_class)
        self.model = self.model.to(device)

        # set up DataLoader for training set
        dataset = Dataset(self.X_, self.y_)
        loader = DataLoader(dataset, shuffle=True, batch_size=256)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_idx, data in enumerate(loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = F.nll_loss(outputs, labels)
                loss.backward()

                optimizer.step()
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_tensor = torch.from_numpy(X).type(torch.FloatTensor)
        X_tensor = X_tensor.to(device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted_labels = torch.argmax(outputs, dim=1)

        return predicted_labels.cpu().numpy()