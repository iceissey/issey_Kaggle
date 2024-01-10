import pandas as pd

"""准备数据"""
input_csv = "../../../archive/train.csv"
data = pd.read_csv(input_csv)
print(len(data))
label_columns = data.columns[-6:]  # 提取labels列
print(label_columns)

data['text'] = data['TITLE'] + " " + data['ABSTRACT']  # 准备text
print(data['text'].head())

data['labels'] = data[label_columns].apply(lambda x: '; '.join(x.index[x == 1]), axis=1)
print(data['labels'])
# Displaying the updated dataset
preprocessed_data = data[['text', 'labels']]
print(preprocessed_data.head())

# 存储为新的 CSV 文件
output_path = "../../../archive/preprocessed_data.csv"
preprocessed_data.to_csv(output_path, index=False)

