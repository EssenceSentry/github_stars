# topic.classification

Automated Topic Classification for News

Performance benchmarks for different models are upon request.

Below are observations of different implemented models:

# Naive Bayes Classifier

Gaussian NB is not suitable in this case, since conditional probabilities are not Gaussian.

# Random Forest

RF makes the split based on randomly selected features which are very sparse, so it yields worse performance than naive Bayes classifiers.

# Boostrap Aggregating (Bagging)

Bagging trees make splits based on all features.

As max_features gets higher, the randomness between trees is reduced and it leads to slightly worse performance than RF (~ 1%).

We also grid-search max_depth for bagging trees and find that performance greatly degraded when max_depth gets lower.

Bagging reduces variance for high-variance models, so it does not improve performance if we apply bagging to low-variance models like naive Bayes classifiers.

# (Stochastic) Gradient Boosting

For gradient boosting, lower subsample (a.k.a., stochastic gradient boosting) prevents from overfitting

It's better to fully grow the trees (consistent with results in bagging trees) and better to have higher n_estimators

If we train gradient boosting machine with max_features = sqrt, performance will get slightly worse, but greatly shorten training time

Number of features does not have hugh impacts on performance as max_depth, n_estimators do

# Linear SVM

Linear SVM with bigram/TFIDF yields the best accuracy (~89%) among all models

# Text Preprocessing

Stop words bear no information here so we remove them

Titles are too short to bear accurate information for classification, so separating titles and content does not work

Including an additional bag-of-nouns in addition to original features does not improve accuracy
