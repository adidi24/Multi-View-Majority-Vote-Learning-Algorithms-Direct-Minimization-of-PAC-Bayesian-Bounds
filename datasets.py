#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:30:46 2023

@author: mehdihennequin
"""
import os
import glob

from sklearn.datasets import make_moons

import pandas as pd
import numpy as np
import re


class Nhanes():
    
    def __init__(self):
        self.datasets = {'df': None, 'exam': None, 'lab': None, 'quest': None}
        self.file = '/NHANES/'
        self.path = os.getcwd()
        self.type = "*.csv"
        self.seqn_ids_to_keep = None
        self.rate_missing_values = 0.99

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
        
        return self.datasets
    
    def clearData(self):
        for views in self.datasets.keys():
            self.datasets[views] = self.datasets[views][self.datasets[views]['seqn'].notna()]
            if self.seqn_ids_to_keep is None:
                self.seqn_ids_to_keep = self.datasets[views]['seqn']
            else:
                self.seqn_ids_to_keep = set(self.seqn_ids_to_keep) & set(self.datasets[views]['seqn'])
                
        for views in self.datasets.keys():
            self.datasets[views] = self.datasets[views][self.datasets[views]['seqn'].isin(self.seqn_ids_to_keep)].sort_values('seqn')                

    def clearMissingValues(self):
        for views in self.datasets.keys():
            for col in self.datasets[views].columns:
                missing_rate = self.datasets[views][col].isna().sum() / len(self.datasets[views].index)
                if missing_rate >= self.rate_missing_values or self.datasets[views][col].nunique(col) == 1:
                    self.datasets[views] = self.datasets[views].drop(col, axis=1)
            self.datasets[views] = self.datasets[views].drop('seqn', axis=1).reset_index(drop=True)
            
    def getData(self, return_list = False, domain_datasets=False):
        
        self.load()
        self.clearData()
        self.clearMissingValues()
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
            else : return self.datasets,self.y.to_numpy()
            

        
    
    
        
        
        
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