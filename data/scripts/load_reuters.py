import argparse
import os
import numpy as np
import pandas as pd

import numpy as np


def clearMissingValues(arrays):
    threshold = 40 

    min_cols = np.min([arr.shape[1] for arr in arrays])
    combined_nan_mask = np.full((arrays[0].shape[0], min_cols), False)
    for i, arr in enumerate(arrays):
        combined_nan_mask += np.isnan(arr[:, :min_cols].astype(float))

    combined_mask = combined_nan_mask.sum(axis=1) <= threshold
    print(combined_mask)
    # for arr in arrays:
    #     arr = arr[combined_mask]


    filtered_arrays = [arr[combined_mask] for arr in arrays]
    return filtered_arrays
        

def load_views(language, views_directory):
    views = []
    
    for filename in os.listdir(views_directory):
        if filename.endswith(f'.csv'):
            file_path = os.path.join(views_directory, filename)
            df = pd.read_csv(file_path)
            df = df.ffill(axis=1) 
            views.append(df.to_numpy())
    return views

def load_dataset(language, base_directory="/Reuters-rcv1-rcv2/"):
    language_directory = os.path.join(os.getcwd() + base_directory, f"ds_{language}")

    # Load all views
    views = load_views(language, language_directory)

    Xs = []
    for v in views:
        Xs.append(v[:, 1:])

    y = views[0][:, 0]

    return Xs, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--base_dir", default="/Reuters-rcv1-rcv2/", type=str, help="The base directory of Reuters RCV1/RCV2 dataset."
    )
    parser.add_argument('--lang', default="EN", type=str,
                        help='Language to load.')

    argv = parser.parse_args()

    Xs, y = load_dataset(argv.lang, argv.base_dir)

    for i, X in enumerate(Xs):
        print(f"View {i + 1} shape: {X.shape}")

    print(f"y shape: {y.shape}")
