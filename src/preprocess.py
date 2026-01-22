import argparse
from pathlib import Path
import torch
import sentencepiece as spm


def read_tsv(path: Path):
    """Yield (zh, en) from TSV lines: zh<TAB>en"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            zh = parts[0].strip()
            en = parts[1].strip()
            if zh and en:
                yield zh, en


def encode_with_bos_eos(sp, text: str, bos_id: int, eos_id: int):
    ids = sp.encode(text, out_type=int)
    return [bos_id] + ids + [eos_id]


def build_examples(sp, tsv_path: Path, max_len: int, direction: str):
    """
    Build examples:
      zh2en: src="<2en> {zh}", tgt="{en}"
      en2zh: src="<2zh> {en}", tgt="{zh}"
      both: create both directions
    Each example stores src_ids and tgt_ids.
    """
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    examples = []
    dropped = 0

    for zh, en in read_tsv(tsv_path):
        pairs = []
        if direction in ("zh2en", "both"):
            pairs.append((f"<2en> {zh}", en))
        if direction in ("en2zh", "both"):
            pairs.append((f"<2zh> {en}", zh))
        for src_text, tgt_text in pairs:
            src_ids = encode_with_bos_eos(sp, src_text, bos_id, eos_id)
            tgt_ids = encode_with_bos_eos(sp, tgt_text, bos_id, eos_id)

            if len(src_ids) > max_len or len(tgt_ids) > max_len:
                dropped += 1
                continue

            examples.append({
                "src_ids": src_ids,
                "tgt_ids": tgt_ids,
            })

    return examples, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm_model", type=str, default="data/spm/zh_en.model")
    ap.add_argument("--input_tsv", type=str, required=True)
    ap.add_argument("--output_pt", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--direction", type=str, default="zh2en",
                    choices=["zh2en", "en2zh", "both"],
                    help="Direction for examples: zh2en, en2zh, or both.")
    args = ap.parse_args()

    spm_model = Path(args.spm_model)
    if not spm_model.exists():
        raise FileNotFoundError(f"Cannot find SentencePiece model: {spm_model}")

    input_tsv = Path(args.input_tsv)
    if not input_tsv.exists():
        raise FileNotFoundError(f"Cannot find input TSV: {input_tsv}")

    out_pt = Path(args.output_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_model))

    examples, dropped = build_examples(sp, input_tsv, max_len=args.max_len, direction=args.direction)

    torch.save({
        "examples": examples,
        "meta": {
            "spm_model": str(spm_model),
            "input_tsv": str(input_tsv),
            "max_len": args.max_len,
            "direction": args.direction,
            "num_examples": len(examples),
            "num_dropped": dropped,
            "pad_id": sp.pad_id(),
            "bos_id": sp.bos_id(),
            "eos_id": sp.eos_id(),
            "unk_id": sp.unk_id(),
        }
    }, out_pt)

    print(f"[OK] Saved tokenized dataset to: {out_pt}")
    print(f"     examples={len(examples)}  dropped={dropped}")


if __name__ == "__main__":
    main()
