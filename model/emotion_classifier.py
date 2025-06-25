import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class EmotionClassifier(nn.Module):
    """
    基于ResNet18的情绪识别模型
    支持7种情绪分类：angry, disgust, fear, happy, neutral, sad, surprise
    """
    
    def __init__(self, num_classes=7, load_pretrained=True):
        super(EmotionClassifier, self).__init__()
        self.num_classes = num_classes
        
        # 根据参数决定是否使用预训练权重
        if load_pretrained:
            try:
                self.backbone = models.resnet18(pretrained=True)
                print("✅ 成功加载预训练的ResNet18模型")
            except Exception as e:
                print(f"⚠️  在线下载失败: {e}")
                # 尝试从本地缓存加载
                try:
                    import torch
                    cache_path = torch.hub.get_dir()
                    model_path = f"{cache_path}/checkpoints/resnet18-f37072fd.pth"
                    if os.path.exists(model_path):
                        print("🔄 尝试从本地缓存加载预训练权重...")
                        self.backbone = models.resnet18(pretrained=False)
                        state_dict = torch.load(model_path, map_location='cpu')
                        self.backbone.load_state_dict(state_dict)
                        print("✅ 成功从本地缓存加载预训练权重！")
                    else:
                        print("📁 本地缓存不存在，使用未预训练模型")
                        self.backbone = models.resnet18(pretrained=False)
                except Exception as e2:
                    print(f"⚠️  本地加载也失败: {e2}")
                    self.backbone = models.resnet18(pretrained=False)
        else:
            self.backbone = models.resnet18(pretrained=False)
            print("🔧 加载已训练模型，跳过预训练权重下载")
        
        # 替换最后的全连接层
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 情绪标签映射
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_labels_cn = ['愤怒', '厌恶', '恐惧', '快乐', '中性', '悲伤', '惊讶']
        
    def forward(self, x):
        return self.backbone(x)
    
    def predict_emotion(self, image_path_or_tensor, device='cpu'):
        """
        预测单张图片的情绪
        
        Args:
            image_path_or_tensor: 图片路径或已处理的tensor
            device: 设备类型
            
        Returns:
            dict: 包含预测结果的字典
        """
        self.eval()
        with torch.no_grad():
            if isinstance(image_path_or_tensor, str):
                # 如果是路径，先加载图片
                image = Image.open(image_path_or_tensor).convert('RGB')
                image_tensor = self.preprocess_image(image).unsqueeze(0).to(device)
            elif hasattr(image_path_or_tensor, 'save'):
                # 如果是PIL Image对象
                image_tensor = self.preprocess_image(image_path_or_tensor).unsqueeze(0).to(device)
            else:
                # 如果已经是tensor
                image_tensor = image_path_or_tensor.to(device)
            
            # 模型推理
            outputs = self.forward(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 获取预测结果
            confidence, predicted = torch.max(probabilities, 1)
            predicted_idx = predicted.item()
            confidence_score = confidence.item()
            
            return {
                'emotion': self.emotion_labels[predicted_idx],
                'emotion_cn': self.emotion_labels_cn[predicted_idx],
                'confidence': confidence_score,
                'all_probabilities': probabilities.cpu().numpy().flatten()
            }
    
    @staticmethod
    def preprocess_image(image):
        """
        图片预处理
        
        Args:
            image: PIL Image对象
            
        Returns:
            torch.Tensor: 预处理后的tensor
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(image)
    
    def get_data_transforms(self):
        """
        获取训练和验证用的数据变换
        
        Returns:
            dict: 包含train和val的transforms
        """
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                # 添加更多增强策略
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                # 在ToTensor之后应用的增强
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        }
        return data_transforms 