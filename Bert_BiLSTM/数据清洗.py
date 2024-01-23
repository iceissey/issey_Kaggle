import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# todo:读取数据
df_train = pd.read_csv('./data/archive/train.csv')
df_test = pd.read_csv('./data/archive/test.csv')
df_val = pd.read_csv('./data/archive/val.csv')

# 输出前5行
# print(df_train.head())
# print(df_train.shape)

# print(df_test.head())
# print(df_test.shape)

# print(df_val.head())
# print(df_val.shape)

# todo: 清洗Train
# 观察数据是否平衡
# print(df_train.label.value_counts())
# print(df_train.label.value_counts() / df_train.shape[0] * 100)
# plt.figure(figsize=(8, 4))
# sns.countplot(x='label', data=df_train)
# plt.show()
# print(df_train[df_train.label > 5.0])
# print(df_train[(df_train.label < -1.1)])
# 丢掉异常数据
df_train.drop(df_train[(df_train.label < -1.1) | (df_train.label > 5)].index, inplace=True, axis=0)
df_train.reset_index(inplace=True, drop=True)
# print(df_train.label.value_counts())
# sns.countplot(x='label', data=df_train)
# plt.show()

# 观察是否有空行
# print(df_train.isnull().sum())
# 删除空行数据
df_train.dropna(axis=0, how='any', inplace=True)
df_train.reset_index(inplace=True, drop=True)
# print(df_train.isnull().sum())

# 查看重复数据
# print(df_train.duplicated().sum())
# print(df_train[df_train.duplicated()==True])
# 删除重复数据
index = df_train[df_train.duplicated() == True].index
df_train.drop(index, axis=0, inplace=True)
df_train.reset_index(inplace=True, drop=True)
# print(df_train.duplicated().sum())

# 我们还需要关心的重复数据是text一样但是label不一样的数据。
# print(df_train['text'].duplicated().sum())
# print(df_train[df_train['text'].duplicated() == True])
# 查看例子
# print(df_train[df_train['text'] == df_train.iloc[856]['text']])
# print(df_train[df_train['text'] == df_train.iloc[3096]['text']])
# 去掉text一样但是label不一样的数据
index = df_train[df_train['text'].duplicated() == True].index
df_train.drop(index, axis=0, inplace=True)
df_train.reset_index(inplace=True, drop=True)
# 检查
# print(df_train['text'].duplicated().sum())  # 0
# print(df_train)
# 检查形状与编号
print("======train-clean======")
print(df_train.tail())
print(df_train.shape)
# 查看text最长长度
print(df_train['text'].str.len().sort_values())
# df_train.to_csv('./data/archive/train_clean.csv', index=None)

# # todo: 清洗test
# # 观察数据是否平衡
# # print(df_test.label.value_counts())
# # print(df_test.label.value_counts() / df_test.shape[0] * 100)
# # plt.figure(figsize=(8, 4))
# # sns.countplot(x='label', data=df_test)
# # plt.show()
# # 观察是否有空行
# # print(df_test.isnull().sum())
# # 删除空行数据
# df_test.dropna(axis=0, how='any', inplace=True)
# df_test.reset_index(inplace=True, drop=True)
# # print(df_test.isnull().sum())
# # 查看重复数据
# # print(df_test.duplicated().sum())
# # print(df_test[df_test.duplicated()==True])
# # 删除重复数据
# index = df_test[df_test.duplicated() == True].index
# df_test.drop(index, axis=0, inplace=True)
# df_test.reset_index(inplace=True, drop=True)
# # print(df_test.duplicated().sum())
# # 重复数据是text一样但是label不一样的数据。
# # print(df_test['text'].duplicated().sum())
# # print(df_test[df_test['text'].duplicated() == True])
# # 查看例子
# # print(df_test[df_test['text'] == df_test.iloc[2046]['text']])
# # print(df_test[df_test['text'] == df_test.iloc[3132]['text']])
# # 去掉text一样但是label不一样的数据
# index = df_test[df_test['text'].duplicated() == True].index
# df_test.drop(index, axis=0, inplace=True)
# df_test.reset_index(inplace=True, drop=True)
# # 检查
# # print(df_test['text'].duplicated().sum())  # 0
# # print(df_test)
# # 检查形状与编号
# print("======test-clean======")
# print(df_test.tail())
# print(df_test.shape)
# # df_test.to_csv('./data/archive/test_clean.csv', index=None)
#
# # todo: 清洗验证集
# # 观察数据是否平衡
# # print(df_val.label.value_counts())
# # print(df_val.label.value_counts() / df_val.shape[0] * 100)
# # plt.figure(figsize=(8, 4))
# # sns.countplot(x='label', data=df_val)
# # plt.show()
# # 丢掉异常数据
# df_val.drop(df_val[(df_val.label == '4') |
#                    (df_val.label == '-') |
#                    (df_val.label == '·')].index, inplace=True, axis=0)
# df_val.reset_index(inplace=True, drop=True)
# # print(df_val.label.value_counts())
# # sns.countplot(x='label', data=df_val)
# # plt.show()
#
# # 观察是否有空行
# # print(df_val.isnull().sum())
# # 删除空行数据
# df_val.dropna(axis=0, how='any', inplace=True)
# df_val.reset_index(inplace=True, drop=True)
# # print(df_val.isnull().sum())
#
# # 查看重复数据
# # print(df_val.duplicated().sum())
# # print(df_val[df_val.duplicated()==True])
# # 删除重复数据
# index = df_val[df_val.duplicated() == True].index
# df_val.drop(index, axis=0, inplace=True)
# df_val.reset_index(inplace=True, drop=True)
# # print(df_val.duplicated().sum())
#
# # 重复数据是text一样但是label不一样的数据。
# print(df_val['text'].duplicated().sum())
# # print(df_val[df_val['text'].duplicated() == True])
# # 查看例子
# # print(df_val[df_val['text'] == df_val.iloc[1817]['text']])
# # print(df_val[df_val['text'] == df_val.iloc[2029]['text']])
# # 去掉text一样但是label不一样的数据
# index = df_val[df_val['text'].duplicated() == True].index
# df_val.drop(index, axis=0, inplace=True)
# df_val.reset_index(inplace=True, drop=True)
# # 检查
# print(df_val['text'].duplicated().sum())  # 0
# # print(df_val)
# # 检查形状与编号
# print("======val-clean======")
# print(df_val.tail())
# print(df_val.shape)
# # df_val.to_csv('./data/archive/val_clean.csv', index=None)