import arff
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import csv
import os
import argparse

# 读取ARFF文件，并转换为DataFrame
def read_arff(file_path):
    """读取ARFF文件并返回DataFrame格式数据。"""
    # 读取ARFF文件
    data = arff.load(open(file_path, 'r'))
    df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
    return df

# DANN模型：包含BERT特征提取器、分类器和域判别器
class DANN(nn.Module):
    # 基于BERT的DANN模型，用于迁移学习
    def __init__(self):
        super(DANN, self).__init__()
        # 使用预训练的BERT模型作为特征提取器
        self.feature_extractor = BertModel.from_pretrained('bert-base-uncased')
        # 定义分类器层
        self.classifier = nn.Sequential(
            nn.Linear(768, 128),  # BERT隐藏层输出维度为768
            nn.ReLU(),
            nn.Linear(128, 2)  # 二分类输出层
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 判断是否来自于源域或目标域
        )

    def forward(self, inputs, alpha=0):
        """前向传播过程，返回分类和域判别输出。"""
        features = self.feature_extractor(**inputs).last_hidden_state[:, 0, :]  # 提取CLS特征
        reverse_features = ReverseLayerF.apply(features, alpha)  # 反向传播中使用梯度反转层
        class_outputs = self.classifier(features)  # 分类输出
        domain_outputs = self.domain_classifier(reverse_features)  # 域判别输出
        return class_outputs, domain_outputs

# 梯度反转层的实现，用于对抗训练
class ReverseLayerF(torch.autograd.Function):
    """梯度反转层，用于对抗训练中的域适应。"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha  # 存储alpha参数，用于梯度反转
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """反转梯度的符号。"""
        output = grad_output.neg() * ctx.alpha
        return output, None

# 数据加载和模型训练
def prepare_data(data, labels, tokenizer, batch_size):
    """将数据转换为张量，并创建Dataloader用于训练和验证。"""
    inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")  # 文本转张量
    labels = torch.tensor(labels)  # 标签转张量
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
    # 将数据集按8:2的比例划分为训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def train_epoch(model, data_loader, optimizer, criterion, alpha):
    """训练一个epoch，并返回平均损失。"""
    model.train()  # 设置模型为训练模型
    total_loss = 0  # 累计损失
    for input_ids, attention_mask, labels in data_loader:
        # 将数据迁移到GPU
        input_ids, attention_mask, labels = input_ids.to(model.device), attention_mask.to(model.device), labels.to(model.device)
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        optimizer.zero_grad()  # 清空梯度
        class_outputs, _ = model(inputs, alpha)  # 前向传播
        loss = criterion(class_outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数
        total_loss += loss.item()  # 累加损失
    return total_loss / len(data_loader)  # 返回平均损失

# 验证或测试过程
def evaluate(model, data_loader):
    """评估模型在验证集上的性能,返回预测结果和真实标签."""
    model.eval()  #  设置模型为评估模式
    predictions, targets = [], []  # 存储预测结果和真实标签
    with torch.no_grad():   # 关闭梯度计算
        for input_ids, attention_mask, labels in data_loader:
            # 将数据迁移到GPU
            input_ids, attention_mask,labels = input_ids.to(model.device), attention_mask.to(model.device), labels.to(model.device)
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            class_outputs, _ = model(inputs)  # 前向传播
            preds = torch.argmax(class_outputs, dim=1).cpu().numpy()  # 获取预测结果
            predictions.extend(preds)
            targets.extend(labels.cpu().numpy())
    return predictions, targets
# 计算常见指标
def compute_metrics(predictions, targets):
    """计算准确率、精确率、召回率和F1分数。"""
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='binary')
    recall = recall_score(targets, predictions, average='binary')
    f1 = f1_score(targets, predictions, average='binary')
    return accuracy, precision, recall, f1

def plot_metrics(train_losses, val_accuracies, val_precisions, val_recalls, val_f1s):
    """绘制损失和各项指标的变化曲线"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_accuracies, 'go-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_precisions, 'ro-', label='Validation Precision')
    plt.title('Validation Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_recalls, 'mo-', label='Validation Recall')
    plt.title('Validation Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.subplot(2, 2, 5)
    plt.plot(epochs, val_f1s, 'mo-', label='Validation F1-Score')
    plt.title('Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_metrics_to_csv(metrics, filename='metrics.csv'):
    """将每个epoch的指标保存为CSV文件"""
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics)

# 主函数：加载数据并进行训练
def main(args):
    """主函数：读取数据、训练模型并进行评估和可视化。"""
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 读取源域和目标域数据
    source_df = read_arff(args.source_file)
    target_df = read_arff(args.target_file)

    # 假设第一列为文本数据
    source_data = source_df.iloc[:, 0].astype(str).tolist()
    target_data = target_df.iloc[:, 0].astype(str).tolist()
    source_labels = source_df.iloc[:, -1].astype(int).tolist()
    target_labels = target_df.iloc[:, -1].astype(int).tolist()

    # 初始化DANN模型、优化器、损失函数和Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #初始化模型并迁移到设备
    model = DANN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 准备数据
    source_train_loader, source_val_loader = prepare_data(source_data, source_labels, tokenizer, args.batch_size)
    target_train_loader, target_val_loader = prepare_data(target_data, target_labels, tokenizer, args.batch_size)

    train_losses = []
    val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], []

    # 训练与验证
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, source_train_loader, optimizer, criterion, alpha=0.1)
        train_losses.append(train_loss)

        val_preds, val_targets = evaluate(model, source_val_loader)
        acc, prec, rec, f1 = compute_metrics(val_preds, val_targets)

        val_accuracies.append(acc)
        val_precisions.append(prec)
        val_recalls.append(rec)
        val_f1s.append(f1)

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {train_loss:.4f}, "
              f"Val Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        save_metrics_to_csv([epoch + 1, train_loss, acc, prec, rec, f1])

    # 测试过程
    print("Testing on target domain...")
    test_preds, test_targets = evaluate(model, target_val_loader)
    acc, prec, rec, f1 = compute_metrics(test_preds, test_targets)
    print(f"Test - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-Score: {f1:.4f}")

    plot_metrics(train_losses, val_accuracies, val_precisions, val_recalls, val_f1s)

# 示例：执行主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DANN with BERT on source and target datasets.')
    parser.add_argument('--source_file', type=str, required=True, help='Path to the source ARFF file.')
    parser.add_argument('--target_file', type=str, required=True, help='Path to the target ARFF file.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')

    args = parser.parse_args()
    main(args)
