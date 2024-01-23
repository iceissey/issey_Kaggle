import pandas as pd
from sklearn.model_selection import train_test_split

# todo: 读取数据
df = pd.read_csv('./data/archive/nCoV_100k_train.labled.csv')
print(df)
# 只要text和标签
df = df[['微博中文内容', '情感倾向']]
df = df.rename(columns={'微博中文内容': 'text', '情感倾向': 'label'})
print(df)

# todo: 分割数据集,储存.0.6/0.2/0.2
train, test = train_test_split(df, test_size=0.2)
train, val = train_test_split(train, test_size=0.25)
print(train)
print(test)
print(val)
train.to_csv('./data/archive/train.csv', index=None)
val.to_csv('./data/archive/val.csv', index=None)
test.to_csv('./data/archive/test.csv', index=None)
