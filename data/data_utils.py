import numpy as np

from .datasets import MultipleFeatures, MNIST_MV_Datasets, CorelImageFeatures

def train_test_merge(Xs_train, y_train, Xs_test, y_test):
    Xs = []
    y = np.concatenate((y_train, y_test))
    for xtr, xts in zip(Xs_train, Xs_test):
        Xs.append(np.concatenate((xtr, xts)))
    return Xs, y


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

def multiclass_to_binary(Xs_train, y_train, Xs_test, y_test, type='ovr', label_1=1, label_2=7):
    if len(np.unique(y_train)) == 2 and len(np.unique(y_test)) == 2:
        print("The dataset is already binary, returning it as is.")
        return Xs_train, y_train, Xs_test, y_test
    
    if type == 'ovr':
        binary_y_train = (y_train == label_1).astype(int)
        binary_y_test = (y_test == label_1).astype(int)
        
        # Balance the binary dataset
        Xs_train_balanced, binary_y_train_balanced = balance_dataset(Xs_train, binary_y_train)
        
        return Xs_train_balanced, binary_y_train_balanced, Xs_test, binary_y_test
    
    elif type == 'ovo':
        # Create a binary y_train for the two classes
        binary_y_train_1 = np.where(y_train == label_1, 1, -1)
        binary_y_train_2 = np.where(y_train == label_2, 1, 0)
        
        merge_train = binary_y_train_1 + binary_y_train_2
        keep_train_indices = np.where(merge_train != -1)
        
        binary_y_train = merge_train[keep_train_indices]
        
        # Create a binary y_test for the two classes
        binary_y_test_1 = np.where(y_test == label_1, 1, -1)
        binary_y_test_2 = np.where(y_test == label_2, 1, 0)
        
        merge_test = binary_y_test_1 + binary_y_test_2
        keep_test_indices = np.where(merge_test != -1)
        
        binary_y_test = merge_test[keep_test_indices]
        
        trimmed_Xs_train = [view[keep_train_indices] for view in Xs_train]
        trimmed_Xs_test = [view[keep_test_indices] for view in Xs_test]
        
        return trimmed_Xs_train, binary_y_train, trimmed_Xs_test, binary_y_test
    
    else:
        raise ValueError("Invalid type. Must be 'ovr' or 'ovo', got {type}")

def balance_dataset(Xs, y):
    # Count the number of positive and negative samples
    num_positive = np.sum(y == 1)
    num_negative = np.sum(y == 0)
    
    minority_label = 1 if num_positive < num_negative else 0
    majority_label = 0 if minority_label == 1 else 1
    
    # Find the indices of the minority and majority class samples
    minority_indices = np.where(y == minority_label)[0]
    majority_indices = np.where(y == majority_label)[0]
    
    # Randomly sample the majority class samples to match the number of minority class samples
    majority_indices_balanced = np.random.choice(majority_indices, size=len(minority_indices), replace=False)
    
    balanced_indices = np.concatenate((minority_indices, majority_indices_balanced))
    np.random.shuffle(balanced_indices)
    
    Xs_balanced = [view[balanced_indices] for view in Xs]
    y_balanced = y[balanced_indices]
    
    return Xs_balanced, y_balanced

def other_binary_options(dataset, y_train, y_test):
    if isinstance(dataset, (MultipleFeatures, MNIST_MV_Datasets)):
        y_train = (y_train % 2 == 0).astype(int)
        y_test = (y_test % 2 == 0).astype(int)
        # y_train = (y_train == (run % 10)).astype(int)
        # y_test = (y_test == (run % 10)).astype(int)
    elif isinstance(dataset, CorelImageFeatures):
        y_train = np.array([1 if value in [2, 3, 6] else 0 for value in y_train])
        y_test = np.array([1 if value in [2, 3, 6] else 0 for value in y_test])
    return y_train,y_test