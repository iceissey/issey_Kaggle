import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import h5py
from tqdm import tqdm
from transformers import XLNetTokenizer, XLNetModel

# 设置设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

"""准备数据"""
# input_csv = "../../archive/train_100kb.csv"
input_csv = "../../archive/train.csv"
data = pd.read_csv(input_csv)  # df
# data = data[:20]  # 测试
print(len(data))
data['combined_text'] = data['TITLE'] + " " + data['ABSTRACT']  # 准备text
# print(data['combined_text'].head())

"""查看单词个数分布情况"""
# 使用空格拆分文本，并计算单词个数
data['word_count'] = data['combined_text'].apply(lambda x: len(str(x).split()))

# 打印单词个数的统计信息
print("单词个数统计信息：")
print("最大单词个数：", data['word_count'].max())
print("最小单词个数：", data['word_count'].min())
print("平均单词个数：", data['word_count'].mean())

# 绘制柱状图,选择并设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用中文黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
plt.figure(figsize=(10, 6))
plt.hist(data['word_count'], bins=50, alpha=0.75, color='b', edgecolor='k')
plt.xlabel('单词个数')
plt.ylabel('频数')
plt.title('单词个数分布')
plt.show()

"""加载XLNet分词器和模型"""
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')
model.to(device)
batch_size = 16  # 确定批处理大小
all_embeddings = []
# token = tokenizer.convert_ids_to_tokens(5)
# print(token)
"""选择不微调嵌入层，于是一次性嵌入所有texts为vector，这样可以大大节省时间"""
texts = data['combined_text'].astype(str).tolist()
for start_index in tqdm(range(0, len(texts[:16]), batch_size)):
    # 对文本进行编码
    batch_texts = texts[start_index:start_index + batch_size]
    print(batch_texts)
    encoded_inputs = tokenizer(batch_texts, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    # 获取嵌入
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    # 计算嵌入
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # 将结果移回CPU并转换为numpy数组
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    # print(embeddings.shape)
    all_embeddings.extend(embeddings)
# 将所有嵌入转换为numpy数组
all_embeddings = np.array(all_embeddings)
print(all_embeddings.shape)
# 存储嵌入向量到HDF5文件
hdf5_filename = '../../../archive/embeddings.h5'
with h5py.File(hdf5_filename, 'w') as hdf5_file:
    hdf5_file.create_dataset('embeddings', data=all_embeddings)

print(f"Embeddings 已存储到 {hdf5_filename} 文件中。")
