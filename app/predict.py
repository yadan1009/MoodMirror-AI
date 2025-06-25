import os
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.emotion_classifier import EmotionClassifier
from app.storage import emotion_storage

class EmotionPredictor:
    """情绪预测器"""
    
    def __init__(self, model_path='model/emotion_model.pt', device=None):
        self.model_path = model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.face_cascade = None
        self.is_loaded = False
        
        # 初始化人脸检测器
        self.init_face_detector()
        
        # 尝试加载模型
        if os.path.exists(model_path):
            self.load_model()
        else:
            print(f"模型文件不存在: {model_path}")
            print("请先训练模型或下载预训练模型")
    
    def init_face_detector(self):
        """初始化OpenCV人脸检测器"""
        try:
            # 尝试加载Haar级联分类器
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("警告: 无法加载人脸检测器")
                self.face_cascade = None
            else:
                print("人脸检测器初始化成功")
                
        except Exception as e:
            print(f"初始化人脸检测器失败: {e}")
            self.face_cascade = None
    
    def load_model(self):
        """加载训练好的模型"""
        try:
            print(f"正在加载模型: {self.model_path}")
            
            # 加载checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 初始化模型（加载已训练模型时不需要预训练权重）
            self.model = EmotionClassifier(num_classes=7, load_pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 获取模型信息
            accuracy = checkpoint.get('accuracy', 0)
            epoch = checkpoint.get('epoch', 0)
            
            print(f"模型加载成功!")
            print(f"训练轮数: {epoch}, 验证准确率: {accuracy:.2f}%")
            
            self.is_loaded = True
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            self.model = None
            self.is_loaded = False
    
    def detect_faces(self, image):
        """
        检测图像中的人脸
        
        Args:
            image: PIL Image对象或numpy数组
            
        Returns:
            list: 检测到的人脸位置列表 [(x, y, w, h), ...]
        """
        if self.face_cascade is None:
            return []
        
        # 转换为OpenCV格式
        if isinstance(image, Image.Image):
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            opencv_image = image
        
        # 转为灰度图
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # 确保返回list格式
        if isinstance(faces, tuple):
            return list(faces) if faces else []
        elif hasattr(faces, 'tolist'):
            return faces.tolist()
        else:
            return list(faces) if len(faces) > 0 else []
    
    def crop_face(self, image, face_coords):
        """
        从图像中裁剪人脸区域
        
        Args:
            image: PIL Image对象
            face_coords: 人脸坐标 (x, y, w, h)
            
        Returns:
            PIL.Image: 裁剪后的人脸图像
        """
        x, y, w, h = face_coords
        
        # 添加一些边距
        margin = 0.2
        mx = int(w * margin)
        my = int(h * margin)
        
        # 计算裁剪区域
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(image.width, x + w + mx)
        y2 = min(image.height, y + h + my)
        
        # 裁剪图像
        face_image = image.crop((x1, y1, x2, y2))
        
        return face_image
    
    def predict_emotion_from_image(self, image, save_to_db=True):
        """
        从图像预测情绪
        
        Args:
            image: PIL Image对象或图像路径
            save_to_db: 是否保存结果到数据库
            
        Returns:
            dict: 预测结果
        """
        if not self.is_loaded:
            # 演示模式：使用随机预测
            print("🎭 演示模式：使用模拟情绪识别")
            return self._demo_prediction(image, save_to_db)
        
        try:
            # 如果是路径，先加载图像
            if isinstance(image, str):
                if not os.path.exists(image):
                    return {'error': f'图像文件不存在: {image}'}
                image = Image.open(image).convert('RGB')
            
            # 检测人脸
            faces = self.detect_faces(image)
            
            if len(faces) == 0:
                # 如果没有检测到人脸，直接使用整张图像
                print("未检测到人脸，使用整张图像进行预测")
                face_image = image
            else:
                # 使用第一个检测到的人脸
                print(f"检测到 {len(faces)} 个人脸，使用第一个人脸")
                face_image = self.crop_face(image, faces[0])
            
            # 预处理图像并使用模型预测
            image_tensor = self.model.preprocess_image(face_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predicted_idx = predicted.item()
                confidence_score = confidence.item()
                
                result = {
                    'emotion': self.model.emotion_labels[predicted_idx],
                    'emotion_cn': self.model.emotion_labels_cn[predicted_idx],
                    'confidence': confidence_score,
                    'all_probabilities': probabilities.cpu().numpy().flatten()
                }
            
            # 添加额外信息
            result['faces_detected'] = len(faces)
            result['model_loaded'] = True
            
            # 保存到数据库
            if save_to_db:
                emotion_storage.add_emotion_record(
                    emotion=result['emotion'],
                    emotion_cn=result['emotion_cn'],
                    confidence=result['confidence'],
                    all_probabilities=result['all_probabilities']
                )
            
            return result
            
        except Exception as e:
            print(f"预测过程中出错: {e}")
            return {
                'error': str(e),
                'emotion': 'error',
                'emotion_cn': '错误',
                'confidence': 0.0
            }
    
    def predict_emotion_from_webcam(self, save_to_db=True):
        """
        从摄像头捕获图像并预测情绪
        
        Args:
            save_to_db: 是否保存结果到数据库
            
        Returns:
            dict: 预测结果
        """
        try:
            # 打开摄像头
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                return {'error': '无法打开摄像头'}
            
            print("按空格键拍照，按ESC键退出")
            
            while True:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    return {'error': '无法读取摄像头画面'}
                
                # 显示画面
                cv2.imshow('情绪识别 - 按空格键拍照', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # 空格键拍照
                if key == ord(' '):
                    # 转换为PIL图像
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    
                    # 预测情绪
                    result = self.predict_emotion_from_image(pil_image, save_to_db)
                    
                    # 显示结果
                    print(f"检测结果: {result['emotion_cn']} (置信度: {result['confidence']:.2f})")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    return result
                
                # ESC键退出
                elif key == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return {'error': '用户取消操作'}
            
        except Exception as e:
            return {'error': f'摄像头操作失败: {e}'}
    
    def batch_predict(self, image_paths, save_to_db=False):
        """
        批量预测图像情绪
        
        Args:
            image_paths: 图像路径列表
            save_to_db: 是否保存结果到数据库
            
        Returns:
            list: 预测结果列表
        """
        results = []
        
        for image_path in image_paths:
            print(f"处理图像: {image_path}")
            result = self.predict_emotion_from_image(image_path, save_to_db)
            result['image_path'] = image_path
            results.append(result)
        
        return results
    
    def get_emotion_emoji(self, emotion):
        """
        根据情绪获取对应的emoji
        
        Args:
            emotion: 英文情绪标签
            
        Returns:
            str: emoji字符
        """
        emoji_map = {
            'happy': '😄',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'neutral': '😐',
            'unknown': '❓',
            'error': '❌'
        }
        
        return emoji_map.get(emotion, '❓')
    
    def _demo_prediction(self, image, save_to_db=True):
        """
        演示模式的情绪预测（使用随机预测）
        
        Args:
            image: PIL Image对象或图像路径
            save_to_db: 是否保存结果到数据库
            
        Returns:
            dict: 模拟预测结果
        """
        import random
        import numpy as np
        
        try:
            # 如果是路径，先加载图像
            if isinstance(image, str):
                if not os.path.exists(image):
                    return {'error': f'图像文件不存在: {image}'}
                from PIL import Image
                image = Image.open(image).convert('RGB')
            
            # 检测人脸
            faces = self.detect_faces(image)
            
            # 模拟情绪预测
            emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            emotions_cn = ['愤怒', '厌恶', '恐惧', '快乐', '中性', '悲伤', '惊讶']
            
            # 生成合理的概率分布（主要情绪概率较高）
            predicted_idx = random.randint(0, 6)
            probabilities = np.random.dirichlet([0.5] * 7)  # 生成和为1的概率分布
            
            # 增强主要情绪的概率
            probabilities[predicted_idx] = max(probabilities[predicted_idx], 0.4 + random.random() * 0.4)
            probabilities = probabilities / probabilities.sum()  # 重新归一化
            
            confidence = probabilities[predicted_idx]
            
            result = {
                'emotion': emotions[predicted_idx],
                'emotion_cn': emotions_cn[predicted_idx],
                'confidence': float(confidence),
                'all_probabilities': probabilities,
                'faces_detected': len(faces),
                'model_loaded': False,
                'demo_mode': True
            }
            
            # 保存到数据库
            if save_to_db:
                emotion_storage.add_emotion_record(
                    emotion=result['emotion'],
                    emotion_cn=result['emotion_cn'],
                    confidence=result['confidence'],
                    all_probabilities=result['all_probabilities']
                )
                print(f"🎭 演示模式预测: {result['emotion_cn']} (置信度: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            print(f"演示预测过程中出错: {e}")
            return {
                'error': str(e),
                'emotion': 'error',
                'emotion_cn': '错误',
                'confidence': 0.0,
                'demo_mode': True
            }

# 创建全局预测器实例
emotion_predictor = EmotionPredictor()

def test_predictor():
    """测试预测器功能"""
    predictor = EmotionPredictor()
    
    if predictor.is_loaded:
        print("模型加载成功，可以开始预测")
        
        # 测试摄像头预测（如果有摄像头的话）
        # result = predictor.predict_emotion_from_webcam()
        # print(f"预测结果: {result}")
    else:
        print("模型未加载，请先训练模型")

if __name__ == '__main__':
    test_predictor() 