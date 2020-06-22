Osprey
======
[![Build Status](https://travis-ci.org/msmbuilder/osprey.svg?branch=master)](https://travis-ci.org/msmbuilder/osprey)
[![Coverage Status](https://coveralls.io/repos/github/msmbuilder/osprey/badge.svg?branch=master)](https://coveralls.io/github/msmbuilder/osprey?branch=master)
[![PyPi version](https://badge.fury.io/py/osprey.svg)](https://pypi.python.org/pypi/osprey/)
[![License](https://img.shields.io/badge/license-ASLv2.0-red.svg?style=flat)](http://www.apache.org/licenses/LICENSE-2.0)
[![DOI](http://joss.theoj.org/papers/10.21105/joss.00034/status.svg)](http://dx.doi.org/10.21105/joss.00034)
[![Research software impact](http://depsy.org/api/package/pypi/osprey/badge.svg)](http://depsy.org/package/python/osprey)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://msmbuilder.org/osprey)

![Logo](http://msmbuilder.org/osprey/development/_static/osprey.svg)

Osprey is an easy-to-use tool for hyperparameter optimization of machine
learning algorithms in Python using scikit-learn (or using scikit-learn
compatible APIs).

Each Osprey experiment combines an dataset, an estimator, a search space
(and engine), cross validation and asynchronous serialization for distributed
parallel optimization of model hyperparameters.

Documentation
------------
For full documentation, please visit the [Osprey homepage](http://msmbuilder.org/osprey/).

Installation
------------

If you have an Anaconda Python distribution, installation is as easy as:
```
$ conda install -c omnia osprey
```

You can also install Osprey with `pip`:
```
$ pip install osprey
```

Alternatively, you can install directly from this GitHub repo:
```
$ git clone https://github.com/msmbuilder/osprey.git
$ cd osprey && git checkout 1.1.0
$ python setup.py install
```


Example using [MSMBuilder](https://github.com/msmbuilder/msmbuilder)
-------------------------------------------------------------
Below is an example of an osprey `config` file to cross validate Markov state
models based on varying the number of clusters and dihedral angles used in a
model:
```yaml
estimator:
  eval_scope: msmbuilder
  eval: |
    Pipeline([
        ('featurizer', DihedralFeaturizer(types=['phi', 'psi'])),
        ('cluster', MiniBatchKMeans()),
        ('msm', MarkovStateModel(n_timescales=5, verbose=False)),
    ])

search_space:
  cluster__n_clusters:
    min: 10
    max: 100
    type: int
  featurizer__types:
    choices:
      - ['phi', 'psi']
      - ['phi', 'psi', 'chi1']
   type: enum

cv: 5

dataset_loader:
  name: mdtraj
  params:
    trajectories: ~/local/msmbuilder/Tutorial/XTC/*/*.xtc
    topology: ~/local/msmbuilder/Tutorial/native.pdb
    stride: 1

trials:
    uri: sqlite:///osprey-trials.db
```

Then run `osprey worker`. You can run multiple parallel instances
of `osprey worker` simultaneously on a cluster too.

```
$ osprey worker config.yaml

...

----------------------------------------------------------------------
Beginning iteration                                              1 / 1
----------------------------------------------------------------------
History contains: 0 trials
Choosing next hyperparameters with random...
  {'cluster__n_clusters': 20, 'featurizer__types': ['phi', 'psi']}

Fitting 5 folds for each of 1 candidates, totalling 5 fits
[Parallel(n_jobs=1)]: Done   1 jobs       | elapsed:    0.3s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    1.8s finished
---------------------------------
Success! Model score = 4.080646
(best score so far   = 4.080646)
---------------------------------

1/1 models fit successfully.
time:         October 27, 2014 10:44 PM
elapsed:      4 seconds.
osprey worker exiting.
```
You can dump the database to JSON or CSV with `osprey dump`.


Dependencies
------------
- `python>=2.7.11`
- `six>=1.10.0`
- `pyyaml>=3.11`
- `numpy>=1.10.4`
- `scipy>=0.17.0`
- `scikit-learn>=0.17.0`
- `sqlalchemy>=1.0.10`
- `bokeh>=0.12.0`
- `matplotlib>=1.5.0`
- `pandas>=0.18.0`
- `GPy` (optional, required for `gp` strategy)
- `hyperopt` (optional, required for `hyperopt_tpe` strategy)
- `nose` (optional, for testing)


Contributing
------------

In case you encounter any issues with this package, please consider submitting
a ticket to the [GitHub Issue Tracker](https://github.com/msmbuilder/osprey/issues).
We also welcome any feature requests and highly encourage users to
[submit pull requests](https://help.github.com/articles/creating-a-pull-request/)
for bug fixes and improvements.

For more detailed information, please refer to our
[documentation](http://msmbuilder.org/osprey/development/contributing.html).


Citing
------

If you use Osprey in your research, please cite:

```bibtex
@misc{osprey,
  author       = {Robert T. McGibbon and
                  Carlos X. Hernández and
                  Matthew P. Harrigan and
                  Steven Kearnes and
                  Mohammad M. Sultan and
                  Stanislaw Jastrzebski and
                  Brooke E. Husic and
                  Vijay S. Pande},
  title        = {Osprey: Hyperparameter Optimization for Machine Learning},
  month        = sep,
  year         = 2016,
  doi          = {10.21105/joss.000341},
  url          = {http://dx.doi.org/10.21105/joss.00034}
}
```
