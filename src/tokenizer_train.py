import argparse
import os
import random
from pathlib import Path

import sentencepiece as spm


def iter_tsv_lines(tsv_path: Path):
    """Yield (src, tgt) from a TSV file: zh<TAB>en"""
    with tsv_path.open("r", encoding="utf-8") as f:
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


def write_corpus_for_spm(
    train_tsv: Path,
    out_txt: Path,
    max_lines: int | None,
    seed: int,
):
    """
    SentencePiece needs a plain text corpus.
    We train a shared vocab on both zh and en by mixing them line-by-line.
    Also include direction tags so the tokenizer learns them.
    """
    random.seed(seed)

    lines_written = 0
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    # If max_lines is set, do a simple reservoir-like sampling by stopping early after shuffle.
    # For 1-10万数据，通常不需要复杂采样：直接全写也行。
    # 这里提供 max_lines 方便你先用小样本跑通。
    buffer = []

    for zh, en in iter_tsv_lines(train_tsv):
        buffer.append(zh)
        buffer.append(en)

    if max_lines is not None and max_lines < len(buffer):
        random.shuffle(buffer)
        buffer = buffer[:max_lines]

    with out_txt.open("w", encoding="utf-8") as out:
        # Direction tokens (for one-model bi-direction)
        out.write("<2en>\n")
        out.write("<2zh>\n")

        for s in buffer:
            out.write(s.replace("\t", " ").strip() + "\n")
            lines_written += 1

    print(f"[OK] Wrote corpus for SPM: {out_txt}  (lines={lines_written})")


def train_sentencepiece(
    corpus_txt: Path,
    model_prefix: Path,
    vocab_size: int,
    character_coverage: float,
):
    """
    Train SentencePiece model.
    We add special tokens (PAD/BOS/EOS/UNK + direction tokens).
    """
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    # We include direction tokens as user_defined_symbols so they are preserved as single tokens.
    user_defined_symbols = "<2en>,<2zh>"

    cmd = (
        f"--input={corpus_txt} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=unigram "
        f"--character_coverage={character_coverage} "
        f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        f"--user_defined_symbols={user_defined_symbols} "
        f"--hard_vocab_limit=false"
    )

    print("[SPM] Training with command:")
    print(cmd)
    spm.SentencePieceTrainer.Train(cmd)

    print(f"[OK] Saved: {model_prefix}.model and {model_prefix}.vocab")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", type=str, default="data/cleaned/train.clean.tsv",
                    help="Path to cleaned train TSV: zh<TAB>en")
    ap.add_argument("--out_corpus", type=str, default="data/spm/spm_corpus.txt",
                    help="Temporary mixed corpus txt for training SentencePiece")
    ap.add_argument("--model_prefix", type=str, default="data/spm/zh_en",
                    help="Output prefix for tokenizer: data/spm/zh_en.model/.vocab")
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument("--max_lines", type=int, default=0,
                    help="If >0, limit total lines for SPM corpus (debug fast). 0 means no limit.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--character_coverage", type=float, default=0.9995,
                    help="For Chinese, usually 0.9995~1.0. For mixed zh/en 0.9995 is safe.")
    args = ap.parse_args()

    train_tsv = Path(args.train_tsv)
    if not train_tsv.exists():
        raise FileNotFoundError(
            f"Cannot find {train_tsv}. Put your cleaned data there, or pass --train_tsv ..."
        )

    out_corpus = Path(args.out_corpus)
    model_prefix = Path(args.model_prefix)

    max_lines = None if args.max_lines <= 0 else args.max_lines

    write_corpus_for_spm(train_tsv, out_corpus, max_lines=max_lines, seed=args.seed)
    train_sentencepiece(
        corpus_txt=out_corpus,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )


if __name__ == "__main__":
    main()
