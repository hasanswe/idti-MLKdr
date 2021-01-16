import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

from sklearn.ensemble import XGBClassifier
from .base import MSVM
from sklearn.linear_model import LogisticRegressio
from sklearn.ensemble.RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix, classification_report, roc_curve, 
                             auc, roc_auc_score, matthews_corrcoef, make_scorer, recall_score, f1_score)


dataset = pd.read_excel(r"D:\Drug Target Interaction\Paper\Experiments\DrugBank.xlsx")

dataset.head()

dataset.Target.value_counts()

dataset.Target.value_counts().plot.bar()

X = dataset.drop("Target", axis=1)
y = dataset["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#
>>> from sklearn.decomposition import TruncatedSVD
>>> from scipy.sparse import random as sparse_random
>>> X = sparse_random(100, 100, density=0.01, format='csr',
...                   random_state=42)
>>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> svd.fit(X)
TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> print(svd.explained_variance_ratio_)
>>> print(svd.explained_variance_ratio_.sum())
>>> print(svd.singular_values_)
#
>>> from sklearn.decomposition import PCA
>>> X = np.array([])
>>> pca = PCA(n_components=2)
>>> pca.fit(X)
PCA(n_components=2)
>>> print(pca.explained_variance_ratio_)
>>> print(pca.singular_values_)
>>> pca = PCA(n_components=2, svd_solver='full')
>>> pca.fit(X)
PCA(n_components=2, svd_solver='full')
>>> print(pca.explained_variance_ratio_)
>>> print(pca.singular_values_)
>>> pca = PCA(n_components=1, svd_solver='arpack')
>>> pca.fit(X)
PCA(n_components=1, svd_solver='arpack')
>>> print(pca.explained_variance_ratio_)
>>> print(pca.singular_values_)
#
>>> from sklearn import random_projection
>>> X = np.random.rand(100, 10000)
>>> transformer = random_projection.GaussianRandomProjection()
>>> X_new = transformer.fit_transform(X)
>>> X_new.shape
#

plt.figure(figsize=(10, 10))
plt.matshow(ranking, cmap="Greens", fignum=1)
plt.colorbar()
plt.title("Ranking", fontsize=18)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
