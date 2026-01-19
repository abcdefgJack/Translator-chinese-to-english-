import sentencepiece as spm
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MTDataset, collate_fn_builder
from model import TransformerMT


def save_ckpt(path: Path, epoch: int, model, optimizer, meta: dict, best_valid: float | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": meta,
            "best_valid": best_valid,
        },
        path,
    )


@torch.no_grad()
def evaluate(model, loader, pad_id: int, device: torch.device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src = batch.src.to(device)
        tgt_in = batch.tgt_in.to(device)
        tgt_out = batch.tgt_out.to(device)

        logits = model(src, tgt_in)  # 评估时不需要传 masks（模型会自己算 key_padding_mask + causal）
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        non_pad = (tgt_out != pad_id).sum().item()
        total_tokens += non_pad
        total_loss += loss.item() * non_pad

    model.train()
    return total_loss / max(1, total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pt", type=str, default="data/bin/train.pt")
    ap.add_argument("--valid_pt", type=str, default="data/bin/valid.pt")
    ap.add_argument("--spm_model", type=str, default="data/spm/zh_en.model")

    ap.add_argument("--ckpt_last", type=str, default="checkpoints/last.pt")
    ap.add_argument("--ckpt_best", type=str, default="checkpoints/best.pt")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--num_threads", type=int, default=0,
                    help="Limit CPU threads for torch (0 = do not override).")
    ap.add_argument("--num_interop_threads", type=int, default=0,
                    help="Limit CPU interop threads for torch (0 = do not override).")

    ap.add_argument("--resume", action="store_true", help="Resume from ckpt_last if it exists.")
    args = ap.parse_args()

    if args.num_threads and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    if args.num_interop_threads and args.num_interop_threads > 0:
        torch.set_num_interop_threads(args.num_interop_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Load tokenizer to get vocab size / special ids -----
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    vocab_size = sp.get_piece_size()

    # ----- Datasets / Loaders -----
    train_ds = MTDataset(args.train_pt)
    valid_ds = MTDataset(args.valid_pt)

    train_collate = collate_fn_builder(train_ds.pad_id)
    valid_collate = collate_fn_builder(valid_ds.pad_id)

    # Windows: num_workers=0 to avoid spawn issues
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_collate,
        num_workers=0,
    )

    # ----- Model -----
    model = TransformerMT(
        vocab_size=vocab_size,
        pad_id=train_ds.pad_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        tie_embeddings=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    # cosine scheduler (per-epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    criterion = nn.CrossEntropyLoss(ignore_index=train_ds.pad_id)

    # ----- Resume -----
    start_epoch = 0
    best_valid = float("inf")
    ckpt_last_path = Path(args.ckpt_last)
    if args.resume and ckpt_last_path.exists():
        try:
            ckpt = torch.load(ckpt_last_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_valid = ckpt.get("best_valid", best_valid)
            print(f"Resumed from {ckpt_last_path} at epoch {start_epoch}, best_valid={best_valid:.4f}")
        except Exception as e:
            print(f"[WARN] Failed to resume from {ckpt_last_path}: {repr(e)}")
            print("Training from scratch.")

    # AMP scaler (newer API avoids deprecation warnings)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ----- Train loop -----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            src = batch.src.to(device)
            tgt_in = batch.tgt_in.to(device)
            tgt_out = batch.tgt_out.to(device)

            # You can pass masks; but model already recomputes what nn.Transformer needs.
            # Keeping your current approach:
            src_pad_mask = batch.src_pad_mask.to(device)
            tgt_pad_mask = batch.tgt_pad_mask.to(device)
            tgt_causal_mask = batch.tgt_causal_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(src, tgt_in, src_pad_mask, tgt_pad_mask, tgt_causal_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            non_pad = (tgt_out != train_ds.pad_id).sum().item()
            total_tokens += non_pad
            total_loss += loss.item() * non_pad

        train_loss = total_loss / max(1, total_tokens)

        # Evaluate on valid
        valid_loss = evaluate(model, valid_loader, pad_id=train_ds.pad_id, device=device)

        # Step scheduler per epoch
        scheduler.step()

        # Print
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss_per_token={train_loss:.4f}  valid_loss_per_token={valid_loss:.4f}  lr={cur_lr:.2e}"
        )

        # Save last
        meta = {
            "vocab_size": vocab_size,
            "spm_model": args.spm_model,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "layers": args.layers,
            "ffn": args.ffn,
            "dropout": args.dropout,
        }
        save_ckpt(ckpt_last_path, epoch, model, optimizer, meta=meta, best_valid=best_valid)
        print(f"Saved checkpoint: {ckpt_last_path}")

        # Save best
        if valid_loss < best_valid:
            best_valid = valid_loss
            ckpt_best_path = Path(args.ckpt_best)
            save_ckpt(ckpt_best_path, epoch, model, optimizer, meta=meta, best_valid=best_valid)
            print(f"Saved checkpoint: {ckpt_best_path} (new best_valid={best_valid:.4f})")

    print("Done.")


if __name__ == "__main__":
    main()
