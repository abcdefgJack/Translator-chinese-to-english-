import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm

from dataset import MTDataset, collate_fn_builder
from model import TransformerMT


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=str, default="data/bin/valid.pt")
    ap.add_argument("--ckpt", type=str, default="checkpoints/last.pt")
    ap.add_argument("--spm_model", type=str, default="data/spm/zh_en.model")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()

    ds = MTDataset(args.pt)
    collate_fn = collate_fn_builder(ds.pad_id)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = TransformerMT(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=256,
        n_heads=4,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        tie_embeddings=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src = batch.src.to(device)
        tgt_in = batch.tgt_in.to(device)
        tgt_out = batch.tgt_out.to(device)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        non_pad = (tgt_out != pad_id).sum().item()
        total_tokens += non_pad
        total_loss += loss.item() * non_pad

    loss_per_token = total_loss / max(1, total_tokens)
    ppl = torch.exp(torch.tensor(loss_per_token)).item()
    print(f"Eval on {args.pt}: loss_per_token={loss_per_token:.4f}, ppl={ppl:.2f}")


if __name__ == "__main__":
    main()
