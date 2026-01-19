import argparse
import random
import re
from pathlib import Path


def looks_like_chinese(s: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", s) is not None


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def read_pairs_tsv(path: Path, swap: bool = False):
    """
    Read TSV pairs from file.

    Expected format:
      - default: zh<TAB>en
      - if swap=True: en<TAB>zh (we swap it to zh<TAB>en)
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            a = normalize_text(parts[0])
            b = normalize_text(parts[1])
            if not a or not b:
                continue

            if swap:
                # input is en<TAB>zh
                zh, en = b, a
            else:
                # input is zh<TAB>en
                zh, en = a, b

            yield zh, en


def clean_pairs(
    pairs,
    min_chars_zh: int = 1,
    min_chars_en: int = 1,
    max_chars: int = 300,
    require_lang_hint: bool = False,
):
    """
    Basic cleaning:
    - length filter by character count (quick and cheap)
    - optional language hint check (off by default):
        zh should contain Chinese; en should NOT contain Chinese (rough heuristic)
    - dedupe by exact pair
    """
    seen = set()
    kept = []
    dropped = {
        "too_short": 0,
        "too_long": 0,
        "lang_mismatch": 0,
        "duplicate": 0,
    }

    for zh, en in pairs:
        if len(zh) < min_chars_zh or len(en) < min_chars_en:
            dropped["too_short"] += 1
            continue

        if len(zh) > max_chars or len(en) > max_chars:
            dropped["too_long"] += 1
            continue

        if require_lang_hint:
            # zh must have Chinese
            if not looks_like_chinese(zh):
                dropped["lang_mismatch"] += 1
                continue
            # en ideally shouldn't contain Chinese (prevents swapped/noisy pairs)
            if looks_like_chinese(en):
                dropped["lang_mismatch"] += 1
                continue

        key = (zh, en)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append((zh, en))

    return kept, dropped


def write_tsv(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for zh, en in pairs:
            f.write(f"{zh}\t{en}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_tsv", type=str, required=True,
                    help="Raw TSV file. Default expects zh<TAB>en.")
    ap.add_argument("--swap", action="store_true",
                    help="If your file is en<TAB>zh, enable this to swap to zh<TAB>en.")
    ap.add_argument("--out_dir", type=str, default="data/cleaned")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.98)
    ap.add_argument("--valid_ratio", type=float, default=0.01)
    ap.add_argument("--test_ratio", type=float, default=0.01)
    ap.add_argument("--max_chars", type=int, default=300)
    ap.add_argument("--lang_check", action="store_true",
                    help="Enable strict language hint checks (off by default).")
    ap.add_argument("--no_lang_check", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must sum to 1.0")

    inp = Path(args.input_tsv)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp}")

    # Read (and swap if needed)
    raw_pairs = list(read_pairs_tsv(inp, swap=args.swap))
    print(f"[1] Loaded raw pairs: {len(raw_pairs)}  (swap={args.swap})")

    require_lang_hint = args.lang_check
    if args.no_lang_check:
        require_lang_hint = False

    kept, dropped = clean_pairs(
        raw_pairs,
        max_chars=args.max_chars,
        require_lang_hint=require_lang_hint,
    )
    print(f"[2] Kept after clean: {len(kept)}")
    print(f"    Dropped stats: {dropped}")

    random.seed(args.seed)
    random.shuffle(kept)

    n = len(kept)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    n_test = n - n_train - n_valid

    train_pairs = kept[:n_train]
    valid_pairs = kept[n_train:n_train + n_valid]
    test_pairs = kept[n_train + n_valid:]

    out_dir = Path(args.out_dir)
    write_tsv(out_dir / "train.clean.tsv", train_pairs)
    write_tsv(out_dir / "valid.clean.tsv", valid_pairs)
    write_tsv(out_dir / "test.clean.tsv", test_pairs)

    print("[3] Wrote:")
    print(f"    train: {len(train_pairs)} -> {out_dir/'train.clean.tsv'}")
    print(f"    valid: {len(valid_pairs)} -> {out_dir/'valid.clean.tsv'}")
    print(f"    test : {len(test_pairs)}  -> {out_dir/'test.clean.tsv'}")


if __name__ == "__main__":
    main()
