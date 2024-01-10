import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, AdamW
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import csv

# 定义超参数
batch_size = 16
epochs = 5
learning_rate = 2e-5


def collate_fn(batch):
    """
    自定义批处理函数
    """
    texts = [item['text'] for item in batch]
    labels = [item['labels'] for item in batch]
    # 使用 tokenizer 对文本和标签进行编码,最大长度512
    encoding = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # 使用 tokenizer 的 target_tokenizer 对标签进行编码
    with tokenizer.as_target_tokenizer():
        labels_encoding = tokenizer(labels, padding=True, truncation=True, max_length=512, return_tensors="pt")
    # 将标签中的 pad token 替换为 -100，这是 T5 模型的要求
    labels_encoding["input_ids"][labels_encoding["input_ids"] == tokenizer.pad_token_id] = -100

    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'labels': labels_encoding['input_ids']
    }


class T5Dataset(Dataset):
    """自定义数据集"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'text': item['text'],
            'labels': item['labels']
        }

    def __len__(self):
        return len(self.dataset)


class T5FineTuner(pl.LightningModule):
    """自定义LightningModule"""

    def __init__(self, train_dataset, val_dataset, test_dataset, learning_rate=2e-5):
        super(T5FineTuner, self).__init__()
        self.validation_loss = []
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.learning_rate = learning_rate  # 微调
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.prediction = []

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return output

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                 shuffle=False)
        return test_loader

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)  # 将loss输出在控制台
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        self.log('val_loss', loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        self.model.eval()
        # 生成输出序列
        generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask)
        # 将生成的token ids转换为文本
        generated_texts = [tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for generated_id in generated_ids]
        # 返回解码后的文本
        # print(generated_texts)
        self.prediction.extend(generated_texts)


def test(model):
    trainer = pl.Trainer(fast_dev_run=False)
    trainer.test(model)
    test_result = model.prediction
    # print(type(test_result))
    for text in test_result[:10]:
        print(text)
    return test_result


def train():
    # 增加回调最优模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控对象为'val_loss'
        dirpath='../../archive/log/T5FineTuner_checkpoints',  # 保存模型的路径
        filename='model-{epoch:02d}-{val_loss:.2f}',  # 最优模型的名称
        save_top_k=1,  # 只保存最好的那个
        mode='min'  # 当监控对象指标最小时
    )
    # 设置日志保存的路径
    log_dir = "../../archive/log"
    logger = TensorBoardLogger(save_dir=log_dir, name="T5FineTuner_logs")
    # Trainer可以帮助调试，比如快速运行、只使用一小部分数据进行测试、完整性检查等，
    # 详情请见官方文档https://lightning.ai/docs/pytorch/latest/debug/debugging_basic.html
    # auto自适应gpu数量
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, accelerator='gpu', devices="auto", fast_dev_run=False,
                         precision=16, callbacks=[checkpoint_callback], logger=logger)
    model = T5FineTuner(train_dataset, valid_dataset, test_dataset, learning_rate)
    trainer.fit(model)
    return model


def save_to_csv(test_dataset, predictions, filename="../../archive/test_predictions.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'true_labels', 'pred_labels'])

        for item, pred_label in zip(test_dataset, predictions):
            text = item['text']
            true_labels = item['labels']
            writer.writerow([text, true_labels, pred_label])


if __name__ == '__main__':
    data = load_dataset('csv', data_files={'train': '../../archive/preprocessed_data.csv'})["train"]
    # 分割数据集为训练集和测试+验证集
    train_testvalid = data.train_test_split(test_size=0.3, seed=42)
    # 分割测试+验证集为测试集和验证集
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    # 现在我们有了训练集、验证集和测试集
    train_dataset = train_testvalid['train']
    valid_dataset = test_valid['train']
    test_dataset = test_valid['test']
    # 打印各个数据集的大小
    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(valid_dataset))
    print("Test set size:", len(test_dataset))
    # 准备Dataset
    train_dataset = T5Dataset(train_dataset)
    valid_dataset = T5Dataset(valid_dataset)
    test_dataset = T5Dataset(test_dataset)
    # print(train_dataset.__len__())
    # print(train_dataset[0])

    # 初始化分词器
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    # 装载dataLoader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  collate_fn=collate_fn, shuffle=True, drop_last=True)
    # 查看装载情况
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i + 1}")
        print("Input IDs:", batch['input_ids'])
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention Mask:", batch['attention_mask'])
        print("Attention Mask shape:", batch['attention_mask'].shape)
        print("Labels:", batch['labels'])
        print("\n")
        if i == 0:
            break

    # model = train()
    model = T5FineTuner.load_from_checkpoint(
        "../../archive/log/T5FineTuner_checkpoints/model-epoch=09-val_loss=0.32.ckpt",
        train_dataset=train_dataset, val_dataset=valid_dataset,
        test_dataset=test_dataset)
    pre_texts = test(model)
    save_to_csv(test_dataset, pre_texts)


