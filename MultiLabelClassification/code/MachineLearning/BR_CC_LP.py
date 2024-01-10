from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

data_path = "../../archive/train.csv"
data = pd.read_csv(data_path)

label_columns = data.columns[-6:]  # 提取labels列
y = data[label_columns].values

# 加载嵌入向量
with h5py.File('../../archive/embeddings.h5', 'r') as f:
    embeddings = np.array(f['embeddings'])

# 确保标签和嵌入向量的行数相同
assert embeddings.shape[0] == y.shape[0]

embeddings = embeddings[:1000]
y = y[:1000]

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=10)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Binary Relevance
start_time = time.time()
br_classifier = BinaryRelevance(RandomForestClassifier())
br_classifier.fit(X_train, y_train)
br_training_time = time.time() - start_time
br_predictions = br_classifier.predict(X_test)
br_precision = precision_score(y_test, br_predictions, average='micro')
br_recall = recall_score(y_test, br_predictions, average='micro')
br_f1 = f1_score(y_test, br_predictions, average='micro')
print("===================================")
print("BR Training Time:", br_training_time)
print("BR Accuracy =", accuracy_score(y_test, br_predictions))
print("BR Precision (micro-average) =", br_precision)
print("BR Recall (micro-average) =", br_recall)
print("BR F1 Score (micro-average) =", br_f1)


# Classifier Chains
start_time = time.time()
cc_classifier = ClassifierChain(RandomForestClassifier())
cc_classifier.fit(X_train, y_train)
cc_training_time = time.time() - start_time
cc_predictions = cc_classifier.predict(X_test)
cc_precision = precision_score(y_test, cc_predictions, average='micro')
cc_recall = recall_score(y_test, cc_predictions, average='micro')
cc_f1 = f1_score(y_test, cc_predictions, average='micro')
print("===================================")
print("CC Training Time:", cc_training_time)
print("CC Accuracy =", accuracy_score(y_test, cc_predictions))
print("CC Precision (micro-average) =", cc_precision)
print("CC Recall (micro-average) =", cc_recall)
print("CC F1 Score (micro-average) =", cc_f1)

# Label Powerset
start_time = time.time()
lp_classifier = LabelPowerset(RandomForestClassifier())
lp_classifier.fit(X_train, y_train)
lp_training_time = time.time() - start_time
lp_predictions = lp_classifier.predict(X_test)
lp_precision = precision_score(y_test, lp_predictions, average='micro')
lp_recall = recall_score(y_test, lp_predictions, average='micro')
lp_f1 = f1_score(y_test, lp_predictions, average='micro')
print("===================================")
print("LP Training Time:", lp_training_time)
print("LP Accuracy =", accuracy_score(y_test, lp_predictions))
print("LP Precision (micro-average) =", lp_precision)
print("LP Recall (micro-average) =", lp_recall)
print("LP F1 Score (micro-average) =", lp_f1)


# 使用SVM作为基础分类器
svm_classifier = OneVsRestClassifier(SVC(kernel='linear'))  # 可以调整kernel和其他SVM参数
# Label Powerset with SVM
start_time = time.time()
lp_svm_classifier = LabelPowerset(svm_classifier)
lp_svm_classifier.fit(X_train, y_train)
lp_svm_training_time = time.time() - start_time
lp_svm_predictions = lp_svm_classifier.predict(X_test)
print("===================================")
print("LP-SVM Training Time:", lp_svm_training_time)
print("LP-SVM Accuracy =", accuracy_score(y_test, lp_svm_predictions))
print("LP-SVM Precision (micro-average) =", precision_score(y_test, lp_svm_predictions, average='micro'))
print("LP-SVM Recall (micro-average) =", recall_score(y_test, lp_svm_predictions, average='micro'))
print("LP-SVM F1 Score (micro-average) =", f1_score(y_test, lp_svm_predictions, average='micro'))
