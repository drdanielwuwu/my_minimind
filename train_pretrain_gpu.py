import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import warnings
import torch
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.MokioModel import MokioMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    setup_seed,
    init_model,
)

warnings.filterwarnings("ignore")

def train_epoch(epoch, loader, iters, start_step=0):
    start_time = time.time()
    current_lr = args.learning_rate  # 初始化学习率
    current_loss = 0.0
    eta_min = 0

    with tqdm(total=iters, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as pbar:
        for step, (input_ids, labels, attention_mask) in enumerate(
            loader, start=start_step + 1
        ):
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            attention_mask = attention_mask.to(args.device)

            lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            current_lr = lr  # 更新当前学习率

            with autocast_ctx:
                res = model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = res.loss + res.aux_loss
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0 or step == iters:
                spend_time = time.time() - start_time
                current_loss = loss.item() * args.accumulation_steps
                eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

                Logger(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
                )

            if (step % args.save_interval == 0 or step == iters):
                model.eval()
                moe_suffix = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
                ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}_gpu.pth"
                state_dict = {k: v.half() for k, v in model.state_dict().items()}
                torch.save(state_dict, ckp)
                model.train()
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{loss.item() * args.accumulation_steps:.4f}",
                "lr": f"{current_lr:.6f}",
                "eta": f"{eta_min}min"
            })

if __name__ == "__main__":
    # GPU 训练配置
    class Args:
        save_dir = "out2"
        save_weight = "pretrain"
        epochs = 1
        batch_size = 32
        learning_rate = 5e-4
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = "bfloat16"
        num_workers = 4
        accumulation_steps = 8
        grad_clip = 1.0
        log_interval = 100
        save_interval = 1000
        hidden_size = 512
        num_hidden_layers = 8
        max_seq_len = 512
        use_moe = 0
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "pretrain_t2t_mini.jsonl")
        from_weight = "none"
        from_resume = 0

    args = Args()

    if not torch.cuda.is_available():
        print("⚠️  未检测到 GPU，自动切换到 CPU 模式")
        args.device = "cpu"
        args.dtype = "float32"
        args.batch_size = 2
        args.num_workers = 0
        args.accumulation_steps = 1

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # 设置混合精度
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        print(f"✅ GPU 开始训练 Epoch {epoch+1} | 批次总数:{len(loader)}")
        train_epoch(epoch, loader, len(loader), 0)
