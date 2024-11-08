# Multi-View Majority Vote Learning Algorithms: Direct Minimization of Oracle PAC-Bayesian Bounds

This repository is the official implementation of ["Multi-View Majority Vote Learning Algorithms: Direct Minimization of Oracle PAC-Bayesian Bounds"]().

The implementation is provided as a module called `mvpb`, which offers an interface for implementing and optimizing PAC-Bayesian bounds specifically designed for multi-view learning.

The `mvpb` module comprises several files and directories, including:

- `multiview_learner.py`: The primary file housing the implementation of the **multi-view** majority vote learning algorithms.

- `bounds/`: A directory containing implementations of the Multi-view PAC-Bayesian `First Order`, `Second Order`, and `C` bounds, found under the `first_order/`, `second_order/`, and `c_bound/` subdirectories, respectively.

- `empirical-evaluation-jupyter.ipynb`: A Jupyter notebook for conducting empirical evaluations and analyses.

## Requirements

To install the required Python packages, you can use the following command:

```sh
pip3 install -r requirements.txt
```

### Downloading the datasets


| Dataset Name      | Original Location | Already Multiview | Number of Views | Number of Samples | Number of Classes | Size    |
|-------------------|-----------------|-------------------|-----------------|-------------------|-------------------|---------|
| aloi_csv        | [ELKI Multi-View Clustering Data Sets Based on ALOI](https://doi.org/10.5281/zenodo.6355684)        | Yes      | 4               | 110250              | 1000                | 673,4 MB   |
| corel_features         | [corel_images (1k)](https://www.kaggle.com/datasets/elkamel/corel-images)        | No      | 7               | 1000              | 10                | 29,9 MB  |
| MNIST_1         | [Multiview Dataset MNIST](https://github.com/goyalanil/Multiview_Dataset_MNIST)        | Yes      | 4               | 70000               | 10                 | 318,7 MB   |
| MNIST_2         | [Multiview Dataset MNIST](https://github.com/goyalanil/Multiview_Dataset_MNIST)        | Yes      | 4               | 70000               | 10                 | 338,3 MB   |
| Fash_MNIST_1         | [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)        | No      | 4               | 70000               | 10                 | 155,6 MB   |
| Fash_MNIST_2         | [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)        | No      | 4               | 70000               | 10                 | 177,6 MB   |
| EMNIST_Letters_1         | [The EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)        | No      | 4               | 70000               | 10                 | 201,1 MB   |
| EMNIST_Letters_2         | [The EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)        | No      | 4               | 70000               | 10                 | 227,7 MB   |
| mfeat         | [Multiple Features](https://archive.ics.uci.edu/dataset/72/multiple+features)        | Yes      | 6               | 2000               | 10                 | 17,5 MB   |
| mfeat-large         | [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/)        | No      | 6               | 70000               | 10                 | 389,5 MB   |
| Mushroom         | [Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)        | No      | 2               | 8124               | 2                 | 0.4 MB   |
| NUS-WIDE-OBJECT         | [NUS-WIDE (LITE)](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)        | Yes      | 5               | 30000               | 31                 | 231,4 MB   |
| PTB-XL-plus         | [PTB-XL+](https://physionet.org/content/ptb-xl-plus/1.0.1/)        | Yes      | 3               | 21800               | 5 Superclasses                 | 248 MB   |
| ReutersEN         | [ReutersEN](http://membres-lig.imag.fr/grimal/)        | Yes      | 5               | 1200               | 6                 | 22,1 MB   |

> Datasets can be pulled using DVC, and the files can be browsed on [DagsHub](https://dagshub.com/adidi24/Multi-View-Majority-Vote-Learning-Algorithms-Direct-Minimization-of-PAC-Bayesian-Bounds) storage.
> Or directly from [OSF](https://osf.io/xh5qs/?view_only=966ab35b04bd4e478491038941f7c141).

For DVC setup, use the following commands:

```sh
dvc remote modify origin --local access_key_id your_token 
dvc remote modify origin --local secret_access_key your_token
```

Then, to pull the datasets, simply execute:

```sh
dvc pull
```

## Running Experiments

Run the cells in the following notebooks:
- empirical-evaluation-jupyter.ipynb to reproduce the main experiments.
- empirical-evaluation-dist.ipynb to reproduce the experiments that compare distributions (before and after adding Gaussian noise to the data).
- plot.ipynb to reproduce all the plots in the paper.

> All our results can also be found in .csv format under the results directory.
> We will move the results to DAGsHub storage later.

## Acknowledgements

The `mvpb` package is slightly inspired by the implementation from <https://github.com/StephanLorenzen/MajorityVoteBounds.git>.

We also took the implementation of the *Deep Neural Decision Forests(dNDF)* in PyTorch from <https://github.com/jingxil/Neural-Decision-Forests.git> with minor modifications.

The inverted-KL optimization code is adapted from <https://github.com/paulviallard/ECML21-PB-CBound>.
