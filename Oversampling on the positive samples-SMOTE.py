numpy as np
import pandas as pd
from sklearn.ensemble import XGBClassifier
from .base import MSVM
from sklearn.linear_model import LogisticRegressio
from sklearn.ensemble.RandomForestClassifier
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix, classification_report, roc_curve, 
                             auc, roc_auc_score, matthews_corrcoef, make_scorer, recall_score, f1_score)
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

dataset = ''  # path to dataset

print("dataset : ", dataset)
df = pd.read_csv(dataset, header=None)

# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# summarize class distribution
counter = Counter(y)
print(counter)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter =  Counter(y)