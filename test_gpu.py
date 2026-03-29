import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from model.MokioModel import MokioMindConfig
from trainer.trainer_utils import init_model

def test_gpu_inference():
    print("🔍 开始 GPU 模型测试")
    
    # 检查GPU是否可用
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  未检测到 GPU，自动切换到 CPU 模式")
    
    # 模型配置
    lm_config = MokioMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        use_moe=False,
    )
    
    # 加载模型
    model, tokenizer = init_model(lm_config, "none", device=device)
    
    # 尝试加载训练好的权重
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./out/pretrain_512_gpu.pth")
    if os.path.exists(model_path):
        print(f"📥 加载权重文件: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("⚠️  未找到训练权重，使用随机初始化模型进行测试")
    
    # 测试推理
    model.eval()
    
    # 输入文本
    test_text = "人工智能是"
    print(f"\n📝 测试输入: {test_text}")
    
    # 编码输入
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    # 生成文本
    print("🤖 生成文本:")
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
    
    # 解码输出
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"{generated_text}")
    
    # 测试批量推理
    print("\n📊 测试批量推理:")
    test_texts = ["今天天气", "机器学习", "Python编程", "深度学习", "自然语言处理"]
    inputs = tokenizer(test_texts, padding=True, return_tensors="pt").to(device)
    
    # 测试推理速度
    import time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            outputs = model(**inputs)
    end_time = time.time()
    
    print(f"批量输入大小: {inputs.input_ids.shape}")
    print(f"输出logits形状: {outputs.logits.shape}")
    print(f"10次推理平均时间: {(end_time - start_time) / 10:.4f}秒")
    
    # 测试混合精度
    if device == "cuda:0":
        print("\n⚡ 测试混合精度推理:")
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                start_time = time.time()
                for _ in range(10):
                    outputs = model(**inputs)
                end_time = time.time()
        print(f"混合精度10次推理平均时间: {(end_time - start_time) / 10:.4f}秒")
    
    print("\n✅ GPU 测试完成!")

if __name__ == "__main__":
    test_gpu_inference()
