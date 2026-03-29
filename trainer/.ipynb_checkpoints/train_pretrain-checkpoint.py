import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import warnings
import torch
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader

from model.MokioModel import MokioMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    lm_checkpoint,
    setup_seed,
    init_model,
)

warnings.filterwarnings("ignore")

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()

    for step, (input_ids, labels, attention_mask) in enumerate(
        loader, start=start_step + 1
    ):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        attention_mask = attention_mask.to(args.device)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

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
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min:"
            )

        if (step % args.save_interval == 0 or step == iters):
            model.eval()
            moe_suffix = "_moe" if hasattr(lm_config, "use_moe") and lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
            state_dict = {k: v.half() for k, v in model.state_dict().items()}
            torch.save(state_dict, ckp)
            model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MokioMind Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", default="pretrain", type=str)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cpu")  # 强制CPU
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", default=0, type=int)
    parser.add_argument("--data_path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset/pretrain_hq.jsonl"))
    parser.add_argument("--from_weight", default="none", type=str)
    parser.add_argument("--from_resume", default=0, type=int)
    args = parser.parse_args()

    setup_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    lm_config = MokioMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
    )

    # CPU 模式
    autocast_ctx = nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        print(f"✅ 开始训练 Epoch {epoch+1} | 批次总数:{len(loader)}")
        train_epoch(epoch, loader, len(loader), 0, None)