import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print(f"正在加载数据: {data_path}")

        # 🔥 核心修复：用 errors='ignore' 跳过所有非法编码
        with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="清洗数据中"):
            line = line.strip()
            if not line:
                continue

            try:
                # 强制清洗字符串
                line = line.encode("utf-8", "ignore").decode("utf-8")
                item = json.loads(line)
                text = item.get("text", "")
                if text:
                    self.samples.append(text)
            except:
                continue

        print(f"✅ 加载完成！有效样本: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return input_ids, input_ids.clone(), attention_mask