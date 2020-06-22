# Recommenders

[![Documentation Status](https://readthedocs.org/projects/microsoft-recommenders/badge/?version=latest)](https://microsoft-recommenders.readthedocs.io/en/latest/?badge=latest)

This repository contains examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learnings on five key tasks: 
- [Prepare Data](notebooks/01_prepare_data): Preparing and loading data for each recommender algorithm
- [Model](notebooks/02_model): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or eXtreme Deep Factorization Machines ([xDeepFM](https://arxiv.org/abs/1803.05170)).
- [Evaluate](notebooks/03_evaluate): Evaluating algorithms with offline metrics
- [Model Select and Optimize](notebooks/04_model_select_and_optimize): Tuning and optimizing hyperparameters for recommender models
- [Operationalize](notebooks/05_operationalize): Operationalizing models in a production environment on Azure

Several utilities are provided in [reco_utils](reco_utils) to support common tasks such as loading datasets in the format expected by different algorithms, evaluating model outputs, and splitting training/test data. Implementations of several state-of-the-art algorithms are included for self-study and customization in your own applications. See the [reco_utils documentation](https://readthedocs.org/projects/microsoft-recommenders/).

For a more detailed overview of the repository, please see the documents at the [wiki page](https://github.com/microsoft/recommenders/wiki/Documents-and-Presentations).

## Getting Started

Please see the [setup guide](SETUP.md) for more details on setting up your machine locally, on a [data science virtual machine (DSVM)](https://azure.microsoft.com/en-gb/services/virtual-machines/data-science-virtual-machines/) or on [Azure Databricks](SETUP.md#setup-guide-for-azure-databricks).

To setup on your local machine:

1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.

2. Clone the repository

```
git clone https://github.com/Microsoft/Recommenders
```

3. Run the generate conda file script to create a conda environment: (This is for a basic python environment, see [SETUP.md](SETUP.md) for PySpark and GPU environment setup)

```
cd Recommenders
python scripts/generate_conda_file.py
conda env create -f reco_base.yaml  
```

4. Activate the conda environment and register it with Jupyter:

```
conda activate reco_base
python -m ipykernel install --user --name reco_base --display-name "Python (reco)"
```

5. Start the Jupyter notebook server

```
jupyter notebook
```

6. Run the [SAR Python CPU MovieLens](notebooks/00_quick_start/sar_movielens.ipynb) notebook under the `00_quick_start` folder. Make sure to change the kernel to "Python (reco)".

**NOTE** - The [Alternating Least Squares (ALS)](notebooks/00_quick_start/als_movielens.ipynb) notebooks require a PySpark environment to run. Please follow the steps in the [setup guide](SETUP.md#dependencies-setup) to run these notebooks in a PySpark environment. For the deep learning algorithms, it is recommended to use a GPU machine.

## Algorithms

The table below lists the recommender algorithms currently available in the repository. Notebooks are linked under the Environment column when different implementations are available.

| Algorithm | Environment | Type | Description |
| --- | --- | --- | --- |
| Alternating Least Squares (ALS) | [PySpark](notebooks/00_quick_start/als_movielens.ipynb) | Collaborative Filtering | Matrix factorization algorithm for explicit or implicit feedback in large datasets, optimized by Spark MLLib for scalability and distributed computing capability |
| Attentive Asynchronous Singular Value Decomposition (A2SVD)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/sequential_recsys_amazondataset.ipynb) | Collaborative Filtering | Sequential-based algorithm that aims to capture both long and short-term user preferences using attention mechanism |
| Cornac/Bayesian Personalized Ranking (BPR) | [Python CPU](notebooks/02_model/cornac_bpr_deep_dive.ipynb) | Collaborative Filtering | Matrix factorization algorithm for predicting item ranking with implicit feedback |
| Convolutional Sequence Embedding Recommendation (Caser) | [Python CPU / Python GPU](notebooks/00_quick_start/sequential_recsys_amazondataset.ipynb) | Collaborative Filtering | Algorithm based on convolutions that aims to capture both user’s general preferences and sequential patterns |
| Deep Knowledge-Aware Network (DKN)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/dkn_synthetic.ipynb) | Content-Based Filtering | Deep learning algorithm incorporating a knowledge graph and article embeddings to provide powerful news or article recommendations |
| Extreme Deep Factorization Machine (xDeepFM)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/xdeepfm_criteo.ipynb) | Hybrid | Deep learning based algorithm for implicit and explicit feedback with user/item features |
| FastAI Embedding Dot Bias (FAST) | [Python CPU / Python GPU](notebooks/00_quick_start/fastai_movielens.ipynb) | Collaborative Filtering | General purpose algorithm with embeddings and biases for users and items |
| LightFM/Hybrid Matrix Factorization | [Python CPU](notebooks/02_model/lightfm_deep_dive.ipynb) | Hybrid | Hybrid matrix factorization algorithm for both implicit and explicit feedbacks |
| LightGBM/Gradient Boosting Tree<sup>*</sup> | [Python CPU](notebooks/00_quick_start/lightgbm_tinycriteo.ipynb) / [PySpark](notebooks/02_model/mmlspark_lightgbm_criteo.ipynb) | Content-Based Filtering | Gradient Boosting Tree algorithm for fast training and low memory usage in content-based problems |
| GRU4Rec | [Python CPU / Python GPU](notebooks/00_quick_start/sequential_recsys_amazondataset.ipynb) | Collaborative Filtering | Sequential-based algorithm that aims to capture both long and short-term user preferences using recurrent neural networks |
| Neural Recommendation with Long- and Short-term User Representations (LSTUR)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/lstur_synthetic.ipynb) | Content-Based Filtering | Neural recommendation algorithm with long- and short-term user interest modeling |
| Neural Recommendation with Attentive Multi-View Learning (NAML)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/naml_synthetic.ipynb) | Content-Based Filtering | Neural recommendation algorithm with attentive multi-view learning |
| Neural Collaborative Filtering (NCF) | [Python CPU / Python GPU](notebooks/00_quick_start/ncf_movielens.ipynb) | Collaborative Filtering | Deep learning algorithm with enhanced performance for implicit feedback |
| Neural Recommendation with Personalized Attention (NPA)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/npa_synthetic.ipynb) | Content-Based Filtering | Neural recommendation algorithm with personalized attention network |
| Neural Recommendation with Multi-Head Self-Attention (NRMS)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/nrms_synthetic.ipynbb) | Content-Based Filtering | Neural recommendation algorithm with multi-head self-attention |
| Restricted Boltzmann Machines (RBM) | [Python CPU / Python GPU](notebooks/00_quick_start/rbm_movielens.ipynb) | Collaborative Filtering | Neural network based algorithm for learning the underlying probability distribution for explicit or implicit feedback |
| Riemannian Low-rank Matrix Completion (RLRMC)<sup>*</sup> | [Python CPU](notebooks/00_quick_start/rlrmc_movielens.ipynb) | Collaborative Filtering | Matrix factorization algorithm using Riemannian conjugate gradients optimization with small memory consumption. |
| Simple Algorithm for Recommendation (SAR)<sup>*</sup> | [Python CPU](notebooks/00_quick_start/sar_movielens.ipynb) | Collaborative Filtering | Similarity-based algorithm for implicit feedback dataset |
| Short-term and Long-term preference Integrated Recommender (SLi-Rec)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/sequential_recsys_amazondataset.ipynb) | Collaborative Filtering | Sequential-based algorithm that aims to capture both long and short-term user preferences using attention mechanism, a time-aware controller and a content-aware controller |
| Surprise/Singular Value Decomposition (SVD) | [Python CPU](notebooks/02_model/surprise_svd_deep_dive.ipynb) | Collaborative Filtering | Matrix factorization algorithm for predicting explicit rating feedback in datasets that are not very large |
| Term Frequency - Inverse Document Frequency (TF-IDF) | [Python CPU](notebooks/00_quick_start/tfidf_covid.ipynb) | Content-Based Filtering | Simple similarity-based algorithm for content-based recommendations with text datasets |
| Vowpal Wabbit Family (VW)<sup>*</sup> | [Python CPU (online training)](notebooks/02_model/vowpal_wabbit_deep_dive.ipynb) | Content-Based Filtering | Fast online learning algorithms, great for scenarios where user features / context are constantly changing |
| Wide and Deep | [Python CPU / Python GPU](notebooks/00_quick_start/wide_deep_movielens.ipynb) | Hybrid | Deep learning algorithm that can memorize feature interactions and generalize user features |
| xLearn/Factorization Machine (FM) & Field-Aware FM (FFM) | [Python CPU](notebooks/02_model/fm_deep_dive.ipynb) | Content-Based Filtering | Quick and memory efficient algorithm to predict labels with user/item features |

**NOTE**: <sup>*</sup> indicates algorithms invented/contributed by Microsoft.

Independent or incubating algorithms and utilities are candidates for the [contrib](contrib) folder. This will house contributions which may not easily fit into the core repository or need time to refactor or mature the code and add necessary tests.

| Algorithm | Environment | Type | Description |
| --- | --- | --- | --- |
| SARplus <sup>*</sup> | [PySpark](contrib/sarplus/README.md) | Collaborative Filtering | Optimized implementation of SAR for Spark |

### Preliminary Comparison

We provide a [benchmark notebook](benchmarks/movielens.ipynb) to illustrate how different algorithms could be evaluated and compared. In this notebook, the MovieLens dataset is split into training/test sets at a 75/25 ratio using a stratified split. A recommendation model is trained using each of the collaborative filtering algorithms below. We utilize empirical parameter values reported in literature [here](http://mymedialite.net/examples/datasets.html). For ranking metrics we use `k=10` (top 10 recommended items). We run the comparison on a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 P100 GPU). Spark ALS is run in local standalone mode. In this table we show the results on Movielens 100k, running the algorithms for 15 epochs.

| Algo | MAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R<sup>2</sup> | Explained Variance | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ALS](notebooks/00_quick_start/als_movielens.ipynb) | 0.004732 |	0.044239 |	0.048462 |	0.017796 | 0.965038 |	0.753001 |	0.255647 |	0.251648 |
| [SVD](notebooks/02_model/surprise_svd_deep_dive.ipynb) | 0.012873	| 0.095930 |	0.091198 |	0.032783 | 0.938681 |	0.742690	| 0.291967 |	0.291971 |
| [SAR](notebooks/00_quick_start/sar_movielens.ipynb) | 0.113028 |	0.388321 | 	0.333828 | 0.183179 | N/A |	N/A |	N/A |	N/A |
| [NCF](notebooks/02_model/ncf_deep_dive.ipynb) | 0.107720	| 0.396118 |	0.347296 |	0.180775 | N/A |	N/A |	N/A |	N/A |
| [BPR](notebooks/02_model/cornac_bpr_deep_dive.ipynb) | 0.105365	| 0.389948 |	0.349841 |	0.181807 | N/A |	N/A |	N/A |	N/A |
| [FastAI](notebooks/00_quick_start/fastai_movielens.ipynb) | 0.025503 |	0.147866 |	0.130329 |	0.053824 | 0.943084 |	0.744337 |	0.285308 |	0.287671 |

## Contributing

This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).

## Build Status

These tests are the nightly builds, which compute the smoke and integration tests. `master` is our main branch and `staging` is our development branch. We use `pytest` for testing python utilities in [reco_utils](reco_utils) and `papermill` for the [notebooks](notebooks). For more information about the testing pipelines, please see the [test documentation](tests/README.md).

### DSVM Build Status

The following tests run on a Windows and Linux DSVM daily. These machines run 24/7.

| Build Type | Branch | Status |  | Branch | Status |
| --- | --- | --- | --- | --- | --- |
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_cpu?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=67&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_cpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=67&branchName=staging) |
| **Linux GPU** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_gpu?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=85&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_gpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=85&branchName=staging) |
| **Linux Spark** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_pyspark?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=86&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_pyspark?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=86&branchName=staging) |
| **Windows CPU** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_cpu?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=101&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_cpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=101&branchName=staging) |
| **Windows GPU** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_gpu?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=102&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_gpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=102&branchName=staging) |
| **Windows Spark** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_pyspark?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=103&branchName=master) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_pyspark?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=103&branchName=staging) |

<!-- Hiding AzureML build status for the moment
### AzureML Build Status

The following tests run on an AzureML [compute target](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-compute-target). AzureML allows to programmatically start a virtual machine, execute the tests, gather the results in [Azure DevOps](https://azure.microsoft.com/en-gb/services/devops/) and shut down the machine.

| Build Type | Branch | Status |  | Branch | Status |  
| --- | --- | --- | --- | --- | --- |  
| **nightly_cpu_tests** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/nightly_cpu_tests?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=25&branchName=master) |   | Staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/nightly_cpu_tests?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=25&branchName=staging) | 
| **nightly_gpu_tests** | master | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/bp-nightly_gpu_tests?branchName=master)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=5&branchName=master) |   | Staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/bp-nightly_gpu_tests?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=5&branchName=staging) |
-->

### Related projects

[Microsoft AI Github](https://github.com/microsoft/ai): Find other Best Practice projects, and Azure AI design patterns in our central repository. 
