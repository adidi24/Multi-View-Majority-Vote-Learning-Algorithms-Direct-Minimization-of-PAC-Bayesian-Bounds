#!/usr/bin/env python3

import csv
import os
import glob
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import Bunch

import pandas as pd
import numpy as np
import scipy.sparse as sp
import re

def train_test_split(Xs, labels, test_size=0.2, random_state=42):

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
        self._name = "Multiple-Features"
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
        

    def get_data(self, **kwargs):
        if self.Xs == [] or self.y == None:
            self.load_data()
        Xs_train, y_train, Xs_test, y_test = train_test_split(self.Xs, self.y)
        return Xs_train, y_train, Xs_test, y_test
    
class SampleData:
    def __init__(self, dataset_path = os.getcwd()+'/data/sample_data/'):
        self._name = "Sample-Data"
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
        

    def get_data(self, **kwargs):
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

    def get_data(self, selected_label=0):
        self.load_data()
        Xs = [self.dataset[X_key] for X_key in self.Xs_filenames]
        y = np.vstack([self.dataset[y_key] for y_key in self.y_filenames]).T
        if y.ndim == 2:
            y = y[:, selected_label]
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
        
class ALOI:
    def __init__(self, dataset_path = os.getcwd()+'/data/aloi_csv/'):
        self._name = "ALOI"
        self.dataset_path = dataset_path
        self.filenames = os.listdir(dataset_path)
        self.filenames.sort()

    def load_data(self):
        dataset = []
        for file in self.filenames:
            df = pd.read_csv(self.dataset_path+file, header=None)
            numerical_df = df.select_dtypes(include=[np.number])
            dataset.append(numerical_df.to_numpy())
        return dataset

    def get_data(self, **kwargs):
        dataset = self.load_data()
        Xs = dataset[:-1]
        y = dataset[-1]
        y = np.argmax(y, axis=1)
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test

class IS:
    def __init__(self):
        self._name = "Image-Segmentation"
        self.le = LabelEncoder()

    def load_data(self):
        from ucimlrepo import fetch_ucirepo 
        image_segmentation = fetch_ucirepo(id=50) 
        self.le.fit(image_segmentation.data.targets)
        labels = self.le.transform(image_segmentation.data.targets)
        return  image_segmentation.data, labels

    def get_data(self, **kwargs):
        dataset, y = self.load_data()
        Xs = [dataset.features.iloc[:, :9].values, dataset.features.iloc[:, 9:].values]
        
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
    def get_real_classes(self, y):
        return self.le.inverse_transform(y)
    
class CorelImageFeatures:
    def __init__(self, dataset_path = os.getcwd()+'/data/corel_features/'):
        self._name = "Corel-Image-Features"
        self.dataset_path = dataset_path
        self.filenames = os.listdir(dataset_path)
        self.le = LabelEncoder()

    def load_data(self):
        dataset = []
        for file in self.filenames:
            if file.endswith('.csv'):
                data = np.loadtxt(self.dataset_path+file, delimiter=',')
                dataset.append(data[:, 1:])
            elif file.endswith('.txt'):
                labels = np.loadtxt(self.dataset_path+file, delimiter=' ', dtype=str)[:, 1]
                self.le.fit(labels)
                labels = self.le.transform(labels)
        return dataset, labels

    def get_data(self, **kwargs):
        Xs, y = self.load_data()
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
    def get_real_classes(self, y):
        return self.le.inverse_transform(y)
    
class ReutersEN:
    def __init__(self, sample=1, select_chi2=1000, dataset_path = os.getcwd()+'/data//ReutersEN/reutersEN/'):
        self._name = "ReutersEN"
        self.dataset_path = dataset_path
        self.sample = sample
        self.views = ['EN', 'FR', 'GR', 'IT', 'SP']
        self.select_chi2 = select_chi2

    def load_data(self):
        data = []
        for view in self.views:
            mtx_file = f"{self.dataset_path}reutersEN_{self.sample}_{view}.mtx"
            maprow_file = f"{self.dataset_path}reutersEN_{self.sample}_{view}.maprow.txt"
            mapcol_file = f"{self.dataset_path}reutersEN_{self.sample}_{view}.mapcol.txt"
            
            with open(mtx_file, 'r') as f:
                # Skip header lines
                for _ in range(2):
                    next(f)

                num_rows, num_cols, num_entries = map(int, next(f).split())

                row_indices = []
                col_indices = []
                data_values = []

                # Read each line in the file and extract row index, column index, and data value
                for line in f:
                    row, col, val = map(float, line.split())
                    row_indices.append(int(row) - 1)
                    col_indices.append(int(col) - 1)
                    data_values.append(val)

                # Construct the sparse matrix
                sparse_mtx = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(num_rows, num_cols))
                dense_array = sparse_mtx.toarray()
                data.append(dense_array)
        return data

    def load_labels(self):
        labels_file = f"{self.dataset_path}labels.txt"
        with open(labels_file, 'r') as f:
            labels = [line.strip() for line in f]
        return labels

    def load_affectations(self):
        act_file = f"{self.dataset_path}reutersEN_act.txt"
        with open(act_file, 'r') as f:
            affectations = [line.strip() for line in f]
        return np.array(affectations, dtype=int)

    def get_data(self, **kwargs):
        Xs = self.load_data()
        y = self.load_affectations()
        if self.select_chi2 is not None:
            for i, X in enumerate(Xs): 
                print(f"Extracting {self.select_chi2} best features by a chi-squared test")
                ch2 = SelectKBest(chi2, k=self.select_chi2)
                Xs[i] = ch2.fit_transform(X, y)
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
class MNIST_MV_Datasets:
    def __init__(self, dataset_path = os.getcwd()+'/data/', sample=1):
        self._name = "MNIST-MV"
        self.dataset_path = dataset_path + f"MNIST_{sample}/"
        self.filenames = os.listdir(self.dataset_path)
        self.filenames.sort()
        self.label_mapping = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
        self.inverted_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def load_data(self):
        dataset = []
        labels = []
        for file in self.filenames:
            view = []
            class_names = []
            with open(self.dataset_path+file, 'r') as f:
                for line in f.readlines():
                    arr = line.split()
                    class_names.append(arr[0])
                    view.append([l.split(':')[1] for l in arr[1:]])
            dataset.append(np.array(view, dtype='int'))
            labels = np.array(class_names, dtype='str')
            
            elabels = np.array([self.label_mapping[label] for label in labels])
        return dataset, elabels

    def get_data(self, **kwargs):
        Xs, y = self.load_data()
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
    def get_real_classes(self, y):
        return np.array([self.inverted_label_mapping[label] for label in y])

class Fash_MNIST_MV_Datasets:
    def __init__(self, dataset_path = os.getcwd()+'/data/', sample=1):
        self._name = "Fashion-MNIST-MV"
        self.dataset_path = dataset_path + f"Fash_MNIST_{sample}/"
        self.filenames = os.listdir(self.dataset_path)
        self.filenames.sort()
        self.label_mapping = {"T-shirt/top": 0, "Trouser": 1, "Pullover": 2, "Pullover": 3, "Coat": 4,
                 "Sandal": 5, "Shirt": 6, "Sneakerv": 7, "Bag": 8, "Ankle boot": 9}
        self.inverted_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def load_data(self):
        dataset = []
        labels = []
        for file in self.filenames:
            view = []
            labels = []
            with open(self.dataset_path+file, 'r') as f:
                for line in f.readlines():
                    arr = line.split()
                    labels.append(arr[0])
                    view.append([val for val in arr[1:]])
            dataset.append(np.array(view, dtype='int'))
            labels = np.array(labels, dtype='int')

        return dataset, labels

    def get_data(self, **kwargs):
        Xs, y = self.load_data()
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
    def get_real_classes(self, y):
        return np.array([self.inverted_label_mapping[label] for label in y])
    
class EMNIST_Letters_MV_Datasets:
    def __init__(self, dataset_path = os.getcwd()+'/data/', sample=1):
        self._name = "EMNIST-Letters-MV"
        self.dataset_path = dataset_path + f"EMNIST_Letters_{sample}/"
        self.filenames = os.listdir(self.dataset_path)
        self.filenames.sort()
        self.label_mapping = {chr(i + 65): i for i in range(26)}
        self.inverted_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def load_data(self):
        dataset = []
        labels = []
        for file in self.filenames:
            view = []
            labels = []
            with open(self.dataset_path+file, 'r') as f:
                for line in f.readlines():
                    arr = line.split()
                    labels.append(arr[0])
                    view.append([val for val in arr[1:]])
            dataset.append(np.array(view, dtype='int'))
            labels = np.array(labels, dtype='int')

        return dataset, labels

    def get_data(self, **kwargs):
        Xs, y = self.load_data()
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
    def get_real_classes(self, y):
        return np.array([self.inverted_label_mapping[label] for label in y])
    
class NUS_WIDE_OBJECT:
    def __init__(self, dataset_path = os.getcwd()+'/data/NUS-WIDE-OBJECT/', min_pos_samples=1000):
        self._name = "NUS-WIDE-OBJECT"
        self.dataset_path = dataset_path
        self.nus_features_files_path = os.listdir(dataset_path+"low level features")
        self.nus_features_files_path.sort()
        self.nus_labels_files_path = os.listdir(dataset_path+"ground truth")
        self.nus_labels_files_path.sort()
        self.min_pos_samples = min_pos_samples
        
        self.views = []

    def load_data(self):
        dataset = []
        y_train, y_test = [], []
        train_set, test_set = [], []

        for file in  self.nus_features_files_path:
            view = file.split('_')[1].split('.')[0]
            with open(f"{self.dataset_path}low level features/{file}", 'r') as f:
                data = []
                for line in f.readlines():
                    data.append([float(x) for x in line.split()])
                if 'Train' in file:
                    train_set.append(np.array(data))
                else:
                    test_set.append(np.array(data))
                self.views.append(view)
        
        for file in self.nus_labels_files_path:
            with open(f"{self.dataset_path}ground truth/{file}", 'r') as f:
                data = []
                for line in f.readlines():
                    data.append(int(line))
                if 'Train' in file:
                    y_train.append(data)
                else:  
                    y_test.append(data)
        
        for xtr, xts in zip(train_set, test_set):
            dataset.append(np.concatenate([xtr[:,:-1], xts[:,:-1]], axis=0))
        labels = np.concatenate([y_train, y_test], axis=1).transpose()
        
        # Remove labels with less than min_pos_samples positive samples
        selected_labels = np.where(labels.sum(axis=0) > self.min_pos_samples)[0]
        labels = labels[:, selected_labels]
        
        return dataset, labels

    def get_data(self, selected_label=0):
        Xs, y = self.load_data()
        if y.ndim == 2:
            y = y[:, selected_label]
        # Xs_balanced, y_balanced = balance_dataset(Xs, y)
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
class Mushrooms:
    def __init__(self, dataset_path = os.getcwd()+'/data/Mushroom/'):
        self._name = "Mushrooms"
        self.dataset_path = dataset_path
        self.filenames = os.listdir(dataset_path)
        self.filenames.sort()

    def load_data(self):
        dataset = []
        for file in self.filenames:
            if file.endswith('.csv'):
                data = pd.read_csv(self.dataset_path+file)
                dataset.append(data.to_numpy())
            else:
                labels = pd.read_csv(self.dataset_path+file).to_numpy().ravel()
        return dataset, labels

    def get_data(self, **kwargs):
        Xs, y = self.load_data()
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
    
class PTB_XL_plus:
    def __init__(self, dataset_path = os.getcwd()+'/data/PTB-XL-plus/'):
        self._name = "PTB-XL+"
        self.dataset_path = dataset_path
        self.filenames = os.listdir(dataset_path)
        self.rate_missing_values = 0.5
        self.le = LabelEncoder()

    def load_data(self):
        dataset = []
        for file in self.filenames:
            if file.endswith('.csv'):
                data = pd.read_csv(self.dataset_path+file, index_col='ecg_id')
                data.sort_values(by='ecg_id', inplace=True)
                dataset.append(data)
            else:
                labels = pd.read_csv(self.dataset_path+file, index_col='ecg_id')
                labels.sort_values(by='ecg_id', inplace=True)

        # Merge all dataframes on the index (ecg_id)
        merged_df = pd.merge(dataset[0], dataset[1], on='ecg_id')
        for df in dataset[2:]:
            merged_df = pd.merge(merged_df, df, on='ecg_id')

        for i, data in enumerate(dataset):
            dataset[i] = data.loc[merged_df.index] #.dropna(axis=1)

        # Extract the labels
        labels = labels.loc[merged_df.index].to_numpy().ravel()
        self.le.fit(labels)
        labels = self.le.transform(labels)
        return dataset, labels

    def clear_missing_values(self, dataset):
        for i, view in enumerate(dataset):
            numerical_cols = view.select_dtypes(include=[np.number, 'float']).columns
            nan_cols = []
            # print(f"View {i}:")
            for col in view.columns:
                missing_rate = view[col].isna().sum() / len(view.index)
                if missing_rate >= self.rate_missing_values or view[col].nunique(col) == 1:
                    # print(f"\t\t {col} with missing rate {missing_rate}")
                    dataset[i] = dataset[i].drop(col, axis=1)
                else:
                    if col in numerical_cols:
                        # print(f"\t\t\t\t {col} with missing rate {missing_rate}")
                        nan_cols.append(col)
                        column = dataset[i][col]
                        dataset[i][col] = column.fillna(column.median())
            dataset[i] = dataset[i].to_numpy()

    def get_data(self, **kwargs):
        Xs, y = self.load_data()
        self.clear_missing_values(Xs)
        Xs_train, y_train, Xs_test, y_test = train_test_split(Xs, y)
        return Xs_train, y_train, Xs_test, y_test
    
    def get_real_classes(self, y):
        return self.le.inverse_transform(y)