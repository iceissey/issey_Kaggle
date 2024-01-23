import torch
from datasets import load_dataset  # hugging-face dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel
import torch.optim as optim
from torch.nn.functional import one_hot


# todo：自定义数据集
class MydataSet(Dataset):
    def __init__(self, path, split):
        self.dataset = load_dataset('csv', data_files=path, split=split)

    def __getitem__(self, item):
        text = self.dataset[item]['text']
        label = self.dataset[item]['label']
        return text, label

    def __len__(self):
        return len(self.dataset)


# todo: 定义批处理函数
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    # 分词并编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,  # 单个句子参与编码
        truncation=True,  # 当句子长度大于max_length时,截断
        padding='max_length',  # 一律补pad到max_length长度
        max_length=200,
        return_tensors='pt',  # 以pytorch的形式返回，可取值tf,pt,np,默认为返回list
        return_length=True,
    )

    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']  # input_ids 就是编码后的词
    attention_mask = data['attention_mask']  # pad的位置是0,其他位置是1
    token_type_ids = data['token_type_ids']  # (如果是一对句子)第一个句子和特殊符号的位置是0,第二个句子的位置是1
    labels = torch.LongTensor(labels)  # 该批次的labels

    # print(data['length'], data['length'].max())
    return input_ids, attention_mask, token_type_ids, labels


# todo: 定义模型，上游使用bert预训练，下游任务选择双向LSTM模型，最后加一个全连接层
class BiLSTM(nn.Module):
    def __init__(self, drop, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.drop = drop
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 加载bert中文模型,生成embedding层
        self.embedding = BertModel.from_pretrained('bert-base-chinese')
        # 预处理模型需要转移至gpu
        self.embedding.to(device)
        # 冻结上游模型参数(不进行预训练模型参数学习)
        for param in self.embedding.parameters():
            param.requires_grad_(False)
        # 生成下游RNN层以及全连接层
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_dim, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=self.drop)
        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)
        # 使用CrossEntropyLoss作为损失函数时，不需要激活。因为实际上CrossEntropyLoss将softmax-log-NLLLoss一并实现的。但是使用其他损失函数时还是需要加入softmax层的。

    def forward(self, input_ids, attention_mask, token_type_ids):
        embedded = self.embedding(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        embedded = embedded.last_hidden_state  # 第0维才是我们需要的embedding,embedding.last_hidden_state = embedding[0]
        out, (h_n, c_n) = self.lstm(embedded)
        output = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        output = self.fc(output)
        return output


if __name__ == '__main__':
    # 设置GPU环境
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device=', device)
    # todo:定义超参数
    batch_size = 128
    epochs = 30
    dropout = 0.4
    rnn_hidden = 768
    rnn_layer = 1
    class_num = 3
    lr = 0.001
    # load train data
    train_dataset = MydataSet('./data/archive/train_clean.csv', 'train')
    # print(train_dataset.__len__())
    # print(train_dataset[0])
    # todo: 加载字典和分词工具
    token = BertTokenizer.from_pretrained('bert-base-chinese')
    # 装载训练集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn,shuffle=True,drop_last=True)
    # 创建模型
    model = BiLSTM(drop=dropout, hidden_dim=rnn_hidden, output_dim=class_num)
    # 模型转移至gpu
    model.to(device)
    # 选择损失函数
    criterion = nn.CrossEntropyLoss()
    # 选择优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # 需要将所有数据转移到gpu
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        labels = labels.long().to(device)
        one_hot_labels = one_hot(labels+1, num_classes=3)
        # 将one_hot_labels类型转换成float
        one_hot_labels = one_hot_labels.to(dtype=torch.float)
        # print(one_hot_labels)
        optimizer.zero_grad()  # 清空梯度
        output = model.forward(input_ids, attention_mask, token_type_ids)  # forward
        # output = output.squeeze()  # 将[128, 1, 3]挤压为[128,3]
        loss = criterion(output, one_hot_labels)  # 计算损失
        print(loss)
        loss.backward()  # backward,计算grad
        optimizer.step()  # 更新参数