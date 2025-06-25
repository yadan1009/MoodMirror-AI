import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# 添加父目录到路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.emotion_classifier import EmotionClassifier

# CutMix数据增强
class CutMix:
    """CutMix数据增强实现"""
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch):
        images, labels = batch
        
        if np.random.rand() > self.prob:
            return images, labels  # 返回原始数据，不应用CutMix
            
        batch_size = images.size(0)
        
        # 生成lambda参数
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机选择两个样本进行混合
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # 生成随机裁剪框
        W = images.size(3)
        H = images.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机选择裁剪位置
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 执行CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images, labels, rand_index, lam

class CutMixCriterion:
    """CutMix损失函数"""
    def __init__(self, criterion):
        self.criterion = criterion
    
    def __call__(self, outputs, labels_a, labels_b, lam):
        return lam * self.criterion(outputs, labels_a) + (1 - lam) * self.criterion(outputs, labels_b)

class EmotionTrainer:
    """情绪识别模型训练器"""
    
    def __init__(self, data_dir, model_save_path, batch_size=32, learning_rate=0.001, num_epochs=20, 
                 resume_from_checkpoint=True, use_cutmix=True, cutmix_alpha=1.0, cutmix_prob=0.5):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.resume_from_checkpoint = resume_from_checkpoint
        self.use_cutmix = use_cutmix
        
        # 检查设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = EmotionClassifier(num_classes=7)
        self.model.to(self.device)
        
        # 准备数据
        self.prepare_data()
        
        # 定义损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.cutmix_criterion = CutMixCriterion(self.criterion)
        
        # 使用AdamW优化器，更好的权重衰减
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                                     weight_decay=0.01, eps=1e-8)
        
        # 使用Cosine Annealing学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        
        # CutMix增强
        if self.use_cutmix:
            self.cutmix = CutMix(alpha=cutmix_alpha, prob=cutmix_prob)
            print(f"✅ 启用CutMix数据增强 (alpha={cutmix_alpha}, prob={cutmix_prob})")
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.start_epoch = 0
        
        # 尝试从checkpoint恢复训练
        self.load_checkpoint()
        
    def prepare_data(self):
        """准备训练和验证数据"""
        data_transforms = self.model.get_data_transforms()
        
        # 加载数据集
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=data_transforms['train']
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'test'),
            transform=data_transforms['val']
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0  # 在Mac上使用0避免多进程问题
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0  # 在Mac上使用0避免多进程问题
        )
        
        # 打印数据集信息
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"类别数: {len(train_dataset.classes)}")
        print(f"类别: {train_dataset.classes}")
        
        # 保存类别标签
        self.class_names = train_dataset.classes
        
    def train_epoch(self):
        """训练一个epoch - 支持CutMix"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='训练中')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # 应用CutMix增强
            if self.use_cutmix:
                cutmix_result = self.cutmix((inputs, labels))
                if len(cutmix_result) == 4:  # CutMix被应用
                    inputs, labels_a, rand_index, lam = cutmix_result
                    labels_b = labels[rand_index]
                    
                    # 前向传播
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.cutmix_criterion(outputs, labels_a, labels_b, lam)
                    
                    # 计算准确率 (CutMix下的近似计算)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (lam * predicted.eq(labels_a).cpu().sum().float() + 
                              (1 - lam) * predicted.eq(labels_b).cpu().sum().float())
                else:
                    # CutMix未被应用，使用正常训练
                    inputs, labels = cutmix_result
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            else:
                # 正常训练（不使用CutMix）
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='验证中')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """完整训练过程"""
        print("开始训练...")
        best_val_acc = max(self.val_accuracies) if self.val_accuracies else 0.0
        
        if self.start_epoch > 0:
            print(f"从epoch {self.start_epoch} 继续训练，当前最佳验证准确率: {best_val_acc:.2f}%")
        
        for epoch in range(self.start_epoch, self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            print('-' * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
            print(f'学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(epoch, val_acc, is_best=True)
                print(f'保存最佳模型! 验证准确率: {val_acc:.2f}%')
        
        print(f'\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%')
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
    def save_model(self, epoch, accuracy, is_best=False):
        """保存模型"""
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': accuracy,
            'class_names': self.class_names,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        
        if is_best:
            torch.save(checkpoint, self.model_save_path)
            print(f"最佳模型已保存到: {self.model_save_path}")
        
        # 也保存最新的模型
        latest_path = self.model_save_path.replace('.pt', '_latest.pt')
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self):
        """从checkpoint恢复训练"""
        if not self.resume_from_checkpoint:
            print("跳过checkpoint恢复，从头开始训练")
            return
            
        # 优先尝试加载最新的模型
        latest_path = self.model_save_path.replace('.pt', '_latest.pt')
        checkpoint_path = None
        
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
            print(f"发现最新checkpoint: {latest_path}")
        elif os.path.exists(self.model_save_path):
            checkpoint_path = self.model_save_path
            print(f"发现最佳checkpoint: {self.model_save_path}")
        else:
            print("未找到checkpoint文件，从头开始训练")
            return
            
        try:
            print(f"正在加载checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态（兼容性处理）
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"⚠️  优化器状态不兼容（从Adam→AdamW）: {e}")
                print("使用新的AdamW优化器状态")
            
            # 加载训练历史
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.val_losses = checkpoint['val_losses']
                self.train_accuracies = checkpoint['train_accuracies']
                self.val_accuracies = checkpoint['val_accuracies']
                
            # 设置起始epoch
            self.start_epoch = checkpoint.get('epoch', 0) + 1
            
            print(f"✅ 成功从checkpoint恢复训练:")
            print(f"   - 已完成epoch: {checkpoint.get('epoch', 0)}")
            print(f"   - 上次验证准确率: {checkpoint.get('accuracy', 0):.2f}%")
            print(f"   - 将从epoch {self.start_epoch} 继续训练")
            
        except Exception as e:
            print(f"❌ 加载checkpoint失败: {e}")
            print("从头开始训练")
            self.start_epoch = 0
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'class_names': self.class_names,
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate
        }
        
        history_path = self.model_save_path.replace('.pt', '_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        print(f"训练历史已保存到: {history_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self):
        # Dummy data for demonstration purposes
        self.train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        self.val_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
        self.train_accuracies = [80, 82, 85, 88, 90]
        self.val_accuracies = [78, 81, 84, 86, 88]
        self.model_save_path = 'model.pt'

    def plot_training_curves(self):
        """Plot training curves"""
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='orange')
        plt.title('Training Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.model_save_path.replace('.pt', '_training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练曲线图已保存到: {plot_path}")


def main():
    """主函数"""
    # 设置路径
    data_dir = 'dataset'  # 数据集目录
    model_save_path = 'model/emotion_model.pt'  # 模型保存路径
    
    print("🚀 启动增强型情绪识别模型训练")
    print("📊 模型架构: ResNet18 (增强版)")
    print("🎯 数据增强: CutMix + 高级增强策略")
    
    # 创建训练器
    trainer = EmotionTrainer(
        data_dir=data_dir,
        model_save_path=model_save_path,
        batch_size=16,              # 减小batch size提高稳定性
        learning_rate=0.0005,       # 降低学习率
        num_epochs=15,              # 增加训练轮数
        resume_from_checkpoint=True,  # 启用断点续训
        use_cutmix=True,            # 启用CutMix
        cutmix_alpha=1.0,           # CutMix参数
        cutmix_prob=0.5             # CutMix概率
    )
    
    # 开始训练
    trainer.train()

if __name__ == '__main__':
    main() 