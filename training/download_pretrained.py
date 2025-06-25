#!/usr/bin/env python3
"""
增强版预训练权重下载器
支持多种下载方式和断点续传
"""

import os
import urllib.request
import ssl
import torch
import requests
import time

def download_with_requests(url, filepath, chunk_size=8192):
    """使用requests库下载，支持断点续传"""
    
    # 检查是否有部分下载的文件
    resume_header = {}
    initial_pos = 0
    if os.path.exists(filepath):
        initial_pos = os.path.getsize(filepath)
        resume_header['Range'] = f'bytes={initial_pos}-'
        print(f"🔄 检测到部分文件，从 {initial_pos} 字节处继续下载")
    
    try:
        # 禁用SSL验证
        session = requests.Session()
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        headers.update(resume_header)
        
        response = session.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('Content-Length', 0)) + initial_pos
        
        mode = 'ab' if initial_pos > 0 else 'wb'
        with open(filepath, mode) as f:
            downloaded = initial_pos
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
        
        print(f"\n✅ 下载完成: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"\n❌ requests下载失败: {e}")
        return None

def download_with_urllib(url, filepath, ssl_verify=False):
    """使用urllib下载"""
    
    try:
        # SSL上下文设置
        if ssl_verify:
            ssl_context = ssl.create_default_context()
        else:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
        
        # 创建请求
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
        
        print(f"🔄 使用urllib下载 (SSL验证: {ssl_verify})...")
        
        with urllib.request.urlopen(request, context=ssl_context, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                chunk_size = 8192
                
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
        
        print(f"\n✅ 下载完成: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"\n❌ urllib下载失败: {e}")
        return None

def download_from_mirrors(filename, filepath):
    """尝试从镜像源下载"""
    
    mirror_urls = [
        f"https://download.pytorch.org/models/{filename}",
        f"https://github.com/pytorch/vision/releases/download/v0.9.0/{filename}",
        f"https://s3.amazonaws.com/pytorch/models/{filename}",
        f"https://download.pytorch.org/models/{filename.replace('-f37072fd', '-5c106cde')}"  # 备用版本
    ]
    
    for i, url in enumerate(mirror_urls, 1):
        print(f"\n🌐 尝试镜像 {i}/{len(mirror_urls)}: {url}")
        
        # 先尝试requests
        try:
            import urllib3
            result = download_with_requests(url, filepath)
            if result:
                return result
        except ImportError:
            print("未安装requests，跳过...")
        except:
            pass
        
        # 再尝试urllib (无SSL验证)
        result = download_with_urllib(url, filepath, ssl_verify=False)
        if result:
            return result
        
        # 最后尝试urllib (有SSL验证)
        result = download_with_urllib(url, filepath, ssl_verify=True)
        if result:
            return result
        
        print(f"❌ 镜像 {i} 失败")
        time.sleep(1)  # 稍等一下再试下一个
    
    return None

def download_resnet18_weights():
    """下载ResNet18预训练权重"""
    
    # 获取PyTorch模型缓存目录
    torch_home = torch.hub.get_dir()
    cache_dir = os.path.join(torch_home, 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    
    # 目标文件信息
    filename = "resnet18-f37072fd.pth"
    filepath = os.path.join(cache_dir, filename)
    expected_size = 46827520  # 大约44.7MB
    
    print(f"📁 缓存目录: {cache_dir}")
    print(f"📄 目标文件: {filename}")
    print(f"💾 预期大小: {expected_size:,} bytes ({expected_size/1024/1024:.1f}MB)")
    
    # 检查文件是否已存在且完整
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if file_size == expected_size:
            print(f"✅ 预训练权重已存在且完整: {filepath}")
            return filepath
        elif file_size > 0:
            print(f"⚠️  检测到不完整文件 ({file_size:,}/{expected_size:,} bytes)")
            choice = input("是否删除并重新下载？(y/n): ")
            if choice.lower() == 'y':
                os.remove(filepath)
                print("🗑️  已删除不完整文件")
        else:
            print(f"🔍 发现空文件，删除重试")
            os.remove(filepath)
    
    print(f"\n📥 开始下载预训练权重...")
    
    # 尝试多种下载方式
    result = download_from_mirrors(filename, filepath)
    
    if result and os.path.exists(result):
        file_size = os.path.getsize(result)
        print(f"\n📊 下载完成！文件大小: {file_size:,} bytes ({file_size/1024/1024:.1f}MB)")
        
        if file_size == expected_size:
            print("✅ 文件大小正确，下载成功！")
        else:
            print(f"⚠️  文件大小不匹配，可能下载不完整")
            print(f"   预期: {expected_size:,} bytes")
            print(f"   实际: {file_size:,} bytes")
        
        return result
    
    return None

def install_requests():
    """安装requests库"""
    try:
        import subprocess
        import sys
        print("📦 正在安装requests库...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "urllib3"])
        print("✅ requests安装成功")
        return True
    except:
        print("❌ requests安装失败")
        return False

if __name__ == "__main__":
    print("🎯 MoodMirror 增强版预训练权重下载器")
    print("=" * 50)
    
    # 检查requests库
    try:
        import requests
        import urllib3
    except ImportError:
        print("⚠️  未检测到requests库，正在安装...")
        if not install_requests():
            print("将使用urllib进行下载...")
    
    result = download_resnet18_weights()
    
    if result:
        print("\n🎉 预训练权重下载成功！")
        print(f"📁 文件位置: {result}")
        print("\n📋 接下来可以:")
        print("1. 重新训练模型获得更好准确率")
        print("2. 运行: python3 training/train_model.py")
    else:
        print("\n😞 所有下载方式都失败了")
        print("\n🔧 建议:")
        print("1. 检查网络连接")
        print("2. 使用VPN或代理")
        print("3. 手动下载文件")
        print("4. 使用优化训练参数继续训练")
