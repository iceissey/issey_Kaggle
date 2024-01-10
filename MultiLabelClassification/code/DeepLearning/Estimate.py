import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np


def convert_labels(label_str):
    return label_str.split(';') if label_str else []


def clean_label(label):
    return label.strip()


# 读取提供的 CSV 文件
file_path = "../../archive/test_predictions.csv"
data = pd.read_csv(file_path)
# print(data.head())

# 提取并转换真实标签和预测标签
true_labels = [convert_labels(label_str) for label_str in data['true_labels']]
pred_labels = [convert_labels(label_str) for label_str in data['pred_labels']]
# 使用清理后的标签重新创建真实标签和预测标签列表
true_labels_cleaned = [list(map(clean_label, label_list)) for label_list in true_labels]
pred_labels_cleaned = [list(map(clean_label, label_list)) for label_list in pred_labels]


# 使用 MultiLabelBinarizer 对标签进行独热编码
mlb = MultiLabelBinarizer()
mlb.fit(true_labels_cleaned + pred_labels_cleaned)
y_true = mlb.transform(true_labels_cleaned)
y_pred = mlb.transform(pred_labels_cleaned)

print("Transformer(T5) Accuracy =", accuracy_score(y_true, y_pred))
print("Transformer(T5) Precision (micro-average) =", precision_score(y_true, y_pred, average='micro'))
print("Transformer(T5) Recall (micro-average) =", recall_score(y_true, y_pred, average='micro'))
print("Transformer(T5) F1 Score (micro-average) =", f1_score(y_true, y_pred, average='micro'))

print("\nAnother way to calculate accuracy:")
# 计算每一列的准确率
column_accuracies = np.mean(y_true == y_pred, axis=0)
# 为每列准确率添加列名
column_accuracy_with_labels = list(zip(mlb.classes_, column_accuracies))
# 计算列准确率的均值
mean_column_accuracy = np.mean(column_accuracies)

for acc in column_accuracy_with_labels:
    print(acc)
# print(column_accuracy_with_labels)
print("Average accuracy = ", mean_column_accuracy)
