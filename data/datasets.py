#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:30:46 2023

@authors: mehdihennequin, abdelkrimzitouni
"""
import csv
import os
import glob
import pickle
from scipy.sparse import csr_matrix

from sklearn.datasets import make_moons

import pandas as pd
import numpy as np
import re

from sklearn.utils import Bunch

def train_test_merge(Xs_train, y_train, Xs_test, y_test):
    Xs = []
    y = np.concatenate((y_train, y_test))
    for xtr, xts in zip(Xs_train, Xs_test):
        Xs.append(np.concatenate((xtr, xts)))
    return Xs, y
    
    
def train_test_split(Xs, labels, test_size=0.3, random_state=42):

    num_views = len(Xs)
    num_samples = len(labels)

    # Shuffle the indices
    indices = np.arange(num_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    # Split data and labels
    split_index = int(num_samples * test_size)
    test_indices, train_indices = indices[:split_index], indices[split_index:]

    Xs_train = [view[train_indices] for view in Xs]
    y_train = labels[train_indices]

    Xs_test = [view[test_indices] for view in Xs]
    y_test = labels[test_indices]

    return Xs_train, y_train, Xs_test, y_test

def s1_s2_split(Xs_train, y_train, Xs_test, y_test, s1_size=0.4, random_state=42):
    num_views = len(Xs_train)
    train_samples = len(y_train)
    test_samples = len(y_test)

    # Shuffle the indices
    train_indices, test_indices = np.arange(train_samples), np.arange(test_samples)
    np.random.seed(random_state)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # Split data and labels
    s1_train_split_index = int(train_samples * s1_size)
    s1_test_split_index = int(test_samples * s1_size)
    s1_train_indices, s2_train_indices = train_indices[:s1_train_split_index], train_indices[s1_train_split_index:]
    s1_test_indices, s2_test_indices = test_indices[:s1_test_split_index], test_indices[s1_test_split_index:]

    # print(f"{s1_train_split_index=}\n {s1_test_split_index=}")
    # print(f"{train_indices.shape=}\n {test_indices.shape=}")
    # print(f"{s1_train_indices.shape=}\n {s2_train_indices.shape=}")
    # print(f"{s1_test_indices.shape=}\n {s2_test_indices.shape=}")
    s1_Xs_train = [view[s1_train_indices] for view in Xs_train]
    s1_y_train = y_train[s1_train_indices]
    s1_Xs_test = [view[s1_test_indices] for view in Xs_test]
    s1_y_test = y_test[s1_test_indices]

    s2_Xs_train = [view[s2_train_indices] for view in Xs_train]
    s2_y_train = y_train[s2_train_indices]
    s2_Xs_test = [view[s2_test_indices] for view in Xs_test]
    s2_y_test = y_test[s2_test_indices]
    
    s1 = {
        "Xs_train":s1_Xs_train,
        "y_train":s1_y_train,
        "Xs_test":s1_Xs_test,
        "y_test":s1_y_test
    }
    
    s2 = {
        "Xs_train":s2_Xs_train,
        "y_train":s2_y_train,
        "Xs_test":s2_Xs_test,
        "y_test":s2_y_test
    }
    
    return s1, s2
    
    

class Nhanes():
    
    def __init__(self):
        self._name = "NHANES"
        self.datasets = {'df': None, 'exam': None, 'lab': None, 'quest': None}
        self.file = '/data/NHANES/'
        self.path = os.getcwd()
        self.type = "*.csv"
        self.seqn_ids_to_keep = None
        self.rate_missing_values = 0.5

    def load(self):
        for views in self.datasets.keys():
            dataset = []
            view_path_files = self.path + self.file + views + self.type
            view_all_files = glob.glob(view_path_files)
            for i in view_all_files: 
                dataset.append(pd.read_csv(i)) 
            self.datasets[views] = pd.concat(dataset) 
            self.datasets[views] = self.datasets[views].loc[:, ~(self.datasets[views].columns.str.contains('^Unnamed')|self.datasets[views].columns.str.contains('^unnamed'))]
        ## Remove rows with NaNs in columns diabete, view quest
        self.datasets['quest'] = self.datasets['quest'][self.datasets['quest']['diabete'].notna()]
        self.datasets['df'] = self.datasets['df'][self.datasets['df']['diabete'].notna()]
        
        return self.datasets
    
    def clear_data(self):
        for views in self.datasets.keys():
            self.datasets[views] = self.datasets[views][self.datasets[views]['seqn'].notna()]
            if self.seqn_ids_to_keep is None:
                self.seqn_ids_to_keep = self.datasets[views]['seqn']
            else:
                self.seqn_ids_to_keep = set(self.seqn_ids_to_keep) & set(self.datasets[views]['seqn'])
                
            self.datasets[views] = self.datasets[views][self.datasets[views]['seqn'].isin(self.seqn_ids_to_keep)].sort_values('seqn')

    def clear_missing_values(self):
        for views in self.datasets.keys():
            numerical_cols = self.datasets[views].select_dtypes(include=['float64']).columns
            nan_cols = []
            for col in self.datasets[views].columns:
                missing_rate = self.datasets[views][col].isna().sum() / len(self.datasets[views].index)
                if missing_rate >= self.rate_missing_values or self.datasets[views][col].nunique(col) == 1:
                    self.datasets[views] = self.datasets[views].drop(col, axis=1)
                else:
                    if col in numerical_cols:
                        nan_cols.append(col)
            self.datasets[views][nan_cols] = self.datasets[views][nan_cols].apply(lambda x: x.fillna(x.median()), axis=0)
            self.datasets[views] = self.datasets[views].drop('seqn', axis=1).reset_index(drop=True)
            
    def get_data(self, return_list = False, domain_datasets=False):
        
        self.load()
        self.clear_data()
        self.clear_missing_values()
        index_diabete_label3 = self.datasets['df'].loc[self.datasets['df']['diabete']==3].index
        index_diabete_label9 = self.datasets['df'].loc[self.datasets['df']['diabete']==9].index

        for views in self.datasets.keys():
            self.datasets[views].drop(index_diabete_label3, inplace = True)
            self.datasets[views].drop(index_diabete_label9, inplace = True)
            self.datasets[views].reset_index(drop=True)
            
            
            
        self.y = self.datasets['df']['diabete'].astype('int32')
        self.y[self.y == 1] = 1
        self.y[self.y == 2] = 0
        # 
        # Drop diabete columns
        columns_data = []
        for views in self.datasets.keys():
            if 'diabete' in self.datasets[views].columns:
                self.datasets[views] = self.datasets[views].drop('diabete', axis=1)
            columns_data.append(self.datasets[views].columns)
                
        if domain_datasets : 
            
            self.X_S0 = {'df': None, 'exam': None, 'lab': None, 'quest': None}
            self.X_S1 = {'df': None, 'exam': None, 'lab': None, 'quest': None}
            idx_domain_0 = self.datasets['df'].loc[self.datasets['df']['sex'] == 0].index
            idx_domain_1 = self.datasets['df'].loc[self.datasets['df']['sex'] == 1].index

            for views in self.datasets.keys():
                self.X_S0[views] = self.datasets[views].loc[idx_domain_0]
                self.X_S1[views] = self.datasets[views].loc[idx_domain_1]
                
            self.y_S0 = self.y.loc[idx_domain_0]
            self.y_S1 = self.y.loc[idx_domain_1]
             
            if return_list : return [i.to_numpy() for i in self.X_S0.values()],[i.to_numpy() for i in self.X_S1.values()],self.y.to_numpy()
            else : return self.datasets,self.y.to_numpy()
            
        else:
            if return_list : return [i.to_numpy() for i in self.datasets.values()],self.y.to_numpy()
            else : 
                Xs_train, y_train, Xs_test, y_test = train_test_split([x.to_numpy() for x in self.datasets.values()], self.y.to_numpy())
                return Xs_train, y_train, Xs_test, y_test

class MultipleFeatures:
    def __init__(self, dataset_path = os.getcwd()+'/data/mfeat/'):
        self._name = "Multiple Features"
        self.dataset_path = dataset_path
        self.feature_sets = ["mfeat-fou", "mfeat-fac", "mfeat-kar", "mfeat-pix", "mfeat-zer", "mfeat-mor"]
        self.Xs = []
        self.y = None

    def load_data(self):
        for feature_set in self.feature_sets:
            csv_path = os.path.join(self.dataset_path, f"{feature_set}.csv")
            if os.path.exists(csv_path):
                view = pd.read_csv(csv_path)
                self.Xs.append(view.iloc[:, :-1].to_numpy())
                self.y = view.iloc[:, -1].to_numpy()
            else:
                raise FileNotFoundError(f"{feature_set}.csv not found in {self.dataset_path}")
        

    def get_data(self):
        if self.Xs == [] or self.y == None:
            self.load_data()
        Xs_train, y_train, Xs_test, y_test = train_test_split(self.Xs, self.y)
        return Xs_train, y_train, Xs_test, y_test
    
class SampleData:
    def __init__(self, dataset_path = os.getcwd()+'/data/sample_data/'):
        self._name = "Sample Data"
        self.dataset_path = dataset_path
        self.views = ['view1', 'view2', 'view3', 'view4']
        self.Xs_train = []
        self.y_train = []
        self.Xs_test = []
        self.y_test = []

    def load_data(self):
        for view in self.views:
            # view-specific training data
            self.Xs_train.append(pickle.load(open(self.dataset_path + view + "/" +
                                                        view + "_X_train.p", "rb"), encoding='latin1'))

            self.y_train.append(pickle.load(open(self.dataset_path + view + "/" +
                                                        view + "_y_train.p", "rb"), encoding='latin1'))

            # view-specific test data
            self.Xs_test.append(pickle.load(open(self.dataset_path + view + "/" +
                                                        view + "_X_test.p", "rb"), encoding='latin1'))
            self.y_test.append(pickle.load(open(self.dataset_path + view + "/" +
                                                        view + "_y_test.p", "rb"), encoding='latin1'))
        

    def get_data(self):
        self.load_data()
        return self.Xs_train, np.array(self.y_train), self.Xs_test, np.array(self.y_test)

class Nutrimouse:
    def __init__(self, dataset_path = os.getcwd()+'/data/nutrimouse/'):
        self._name = "Nutrimouse"
        self.dataset_path = dataset_path
        self.dataset = Bunch()
        self.Xs_filenames = ["gene", "lipid"]
        self.y_filenames = ["genotype", "diet"]

    def load_data(self):
        
        for fname in self.Xs_filenames:
            csv_file = os.path.join( self.dataset_path, fname + '.csv')
            X = np.genfromtxt(csv_file, delimiter=',', names=True)
            self.dataset[fname] = X.view((float, len(X.dtype.names)))
            self.dataset[f'{fname}_feature_names'] = list(X.dtype.names)

        for fname in self.y_filenames:
            csv_file = os.path.join( self.dataset_path, fname + '.csv')
            with open(csv_file, newline='') as f:
                y = np.asarray(list(csv.reader(f))[1:]).squeeze()
            class_names, y = np.unique(y, return_inverse=True)
            self.dataset[fname] = y
            self.dataset[f'{fname}_names'] = class_names

    def get_data(self):
        self.load_data()
        Xs = [self.dataset[X_key] for X_key in self.Xs_filenames]
        y = np.vstack([self.dataset[y_key] for y_key in self.y_filenames]).T
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
        
            

        
    
    
        
        
        
# def make_moons_da(n_samples=100, rotation=0, noise=0.05, random_state=0):
#     Xs, ys = make_moons(n_samples=n_samples,
#                         noise=noise,
#                         random_state=random_state)
#     Xs[:, 0] -= 0.5
#     theta = np.radians(-rotation)
#     cos_theta, sin_theta = np.cos(theta), np.sin(theta)
#     rot_matrix = np.array(
#         ((cos_theta, -sin_theta),
#           (sin_theta, cos_theta))
#     )
#     Xt = Xs.dot(rot_matrix)
#     yt = ys
#     return Xs, ys, Xt, yt        

# Xs, ys, Xt, yt = make_moons_da()

# x_min, y_min = np.min([Xs.min(0), Xt.min(0)], 0)
# x_max, y_max = np.max([Xs.max(0), Xt.max(0)], 0)
# x_grid, y_grid = np.meshgrid(np.linspace(x_min-0.1, x_max+0.1, 100),
#                              np.linspace(y_min-0.1, y_max+0.1, 100))
# X_grid = np.stack([x_grid.ravel(), y_grid.ravel()], -1)

# fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
# ax1.set_title("Input space")
# ax1.scatter(Xs[ys==0, 0], Xs[ys==0, 1], label="source", edgecolors='k', c="red")
# ax1.scatter(Xs[ys==1, 0], Xs[ys==1, 1], label="source", edgecolors='k', c="blue")
# ax1.scatter(Xt[:, 0], Xt[:, 1], label="target", edgecolors='k', c="black")

# ax1.legend(loc="lower right")
# ax1.set_yticklabels([])
# ax1.set_xticklabels([])
# ax1.tick_params(direction ='in')
# plt.show()

# datasets  = Nhanes() # load Nhanes dataset
# X,y = datasets.get_data(return_list = True, domain_datasets=False)
# =====
# datasets  = SampleData()
# X, y_train, Xs_test, y_test = datasets.get_data()
# print("number of views",len(X))

# #concatenation of te views###############################
# # X_concat = np.concatenate((X[0],X[1],X[2],X[3]), axis=1)
# ###############################

# X_view1 = X[0]
# X_view2 = X[1]
# X_view3 = X[2]
# X_view4 = X[3]

# for xv in X:
#     print(xv)
    
    
    
    
    