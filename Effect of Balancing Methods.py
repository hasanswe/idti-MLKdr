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

dataset2 = pd.read_excel(r"D:\Drug Target Interaction\Datasets\Main datasets\Experiments\All Dataset\Experiments\Enzyme Drug Target Pair Datasets\DrugBank.xlsx")

dataset2.head()

dataset2.shape

dataset2.Target.value_counts()

dataset2.Target.value_counts().plot.bar()

X = dataset2.drop("Target", axis=1)
y = dataset2["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

clf_msvm2 = MSVM (n_estimators = 200, max_depth = 10, random_state=1)
pred2 = clf_msvm2.fit(X_train, y_train).predict(X_test)
accuracy_score(pred2, y_test)

msvm_pred_proba2 = clf_msvm.predict_proba(X_test)
msvm_pred_proba2 = msvm_pred_proba2[:, 1]

cm_2 = confusion_matrix(pred2, y_test)
cm_2

true_positive = cm_2[0][0]
false_positive = cm_2[0][1]
false_negative = cm_2[1][0]
true_negative = cm_2[1][1]

true_positive, false_positive, false_negative, true_negative

print(classification_report(pred2, y_test))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity

# calculate AUC
auc_msvm_2 = roc_auc_score(y_test, msvm_pred_proba2)
print('AUC value of msvm: %.4f' % auc_msvm_2)

f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure

matthews_corrcoef(pred2, y_test)

# # Cluster Under Sampling (CUS)

CUS= CUS(random_state=10)
X_res, y_res = smote.fit_sample(X,y)

X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_res, y_res, test_size = 0.2, random_state = 1)


pd.Series(y_res).value_counts().plot.bar()


clf_msvm_a = MSVM (n_estimators = 200, max_depth=10, random_state=1)
pred2_a = clf_msvm_a.fit(X_train_a, y_train_a).predict(X_test_a)
accuracy_score(pred2_a, y_test_a)

msvm_pred_proba2_a = clf_msvm_a.predict_proba(X_test_a)
msvm_pred_proba2_a = msvm_pred_proba2_a[:, 1]

cm_2_a = confusion_matrix(pred2_a, y_test_a)
cm_2_a

true_positive = cm_2_a[0][0]
false_positive = cm_2_a[0][1]
false_negative = cm_2_a[1][0]
true_negative = cm_2_a[1][1]

true_positive, false_positive, false_negative, true_negative

print(classification_report(pred2_a, y_test_a))

sensitivity = (true_positive / (true_positive + false_negative))
specificity = (true_negative / (true_negative + false_positive))
sensitivity, specificity

# calculate AUC
auc_msvm_2_a = roc_auc_score(y_test_a, msvm_pred_proba2_a)
print('AUC value of MSVM: %.4f' % auc_msvm_2_a)

f_measure = ((2 * true_positive)/ ((2 * true_positive) + false_positive + false_negative))
f_measure

matthews_corrcoef(pred2_a, y_test_a)


# RUS (Version 1), SMOTE(Version 2) and CluMS (Version 3)

fpr_2, tpr_2, _ = roc_curve(y_test, msvm_pred_proba2)
fpr_2, tpr_2, _ = roc_curve(y_test, msvm_pred_proba2)
fpr_2_a, tpr_2_a, _ = roc_curve(y_test_a, msvm_pred_proba2_a)

roc_auc = dict()

roc_auc[0] = auc(fpr_2, tpr_2)
roc_auc[1] = auc(fpr_2_a, tpr_2_a)
roc_auc

sns.set_style("white")
plt.figure(figsize=(8, 5))
plt.figure(dpi=600)

plt.plot(fpr_2, tpr_2, color = "red" , label = "Version 1 - %0.2f" % roc_auc[0], lw=3)
plt.plot(fpr_2, tpr_2, color = "red" , label = "Version 2 - %0.2f" % roc_auc[0], lw=3)
plt.plot(fpr_2_a, tpr_2_a, color = "blue" , label = "Version 3 - %0.2f" % roc_auc[1], lw=3)

plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('DrugBank- [A, X, Y, Z]', fontsize=22)
plt.legend(loc="lower right", fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()






