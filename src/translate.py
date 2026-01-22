import argparse
import re
import torch
import sentencepiece as spm

from model import TransformerMT


def looks_like_chinese(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


@torch.no_grad()
def beam_search(
    model,
    sp,
    src_ids,
    pad_id,
    bos_id,
    eos_id,
    device,
    max_new_tokens=64,
    num_beams=4,
    length_penalty=1.2,   # 默认更合理：避免过短
    no_repeat_ngram=3,
    repeat_penalty=0.8,
    eos_boost_after=20,
    eos_boost=0.3,
):
    model.eval()

    beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0)]  # (tgt_seq, logprob)

    for _ in range(max_new_tokens):
        new_beams = []

        for tgt_seq, score in beams:
            if tgt_seq[0, -1].item() == eos_id:
                new_beams.append((tgt_seq, score))
                continue

            logits = model(src_ids, tgt_seq)  # (1, T, V)
            next_logits = logits[:, -1, :]    # (1, V)
            log_probs = torch.log_softmax(next_logits, dim=-1).squeeze(0)  # (V,)

            # (A) Repetition penalty: punish repeating the last token
            last_token = tgt_seq[0, -1].item()
            log_probs[last_token] -= float(repeat_penalty)

            # (B) Encourage EOS after some length
            if tgt_seq.size(1) >= int(eos_boost_after):
                log_probs[eos_id] += float(eos_boost)

            # (C) no-repeat ngram (default 3-gram), only check top candidates for speed
            if no_repeat_ngram and tgt_seq.size(1) >= no_repeat_ngram:
                seq = tgt_seq.squeeze(0).tolist()

                # build existing ngrams
                n = int(no_repeat_ngram)
                seen = set()
                for i in range(len(seq) - n + 1):
                    seen.add(tuple(seq[i:i + n]))

                prefix = tuple(seq[-(n - 1):]) if n > 1 else tuple()

                # only ban within top-k candidates (fast)
                cand_k = min(200, log_probs.numel())
                cand = torch.topk(log_probs, k=cand_k).indices.tolist()
                for t in cand:
                    if n == 1:
                        # no-repeat unigram is too strict; usually not used
                        continue
                    cand_ng = prefix + (t,)
                    if cand_ng in seen:
                        log_probs[t] = -1e9

            topk = torch.topk(log_probs, k=num_beams)
            for token_id, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_seq = torch.cat([tgt_seq, torch.tensor([[token_id]], device=device)], dim=1)
                new_beams.append((new_seq, score + lp))

        def norm_score(item):
            seq, sc = item
            L = seq.size(1)
            return sc / ((L ** length_penalty) if length_penalty != 0 else 1.0)

        new_beams.sort(key=norm_score, reverse=True)
        beams = new_beams[:num_beams]

        if all(seq[0, -1].item() == eos_id for seq, _ in beams):
            break

    best_seq, _ = beams[0]
    return best_seq.squeeze(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/last.pt")
    ap.add_argument("--spm_model", type=str, default="data/spm/zh_en.model")
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--len_penalty", type=float, default=1.2)
    ap.add_argument("--no_repeat_ngram", type=int, default=3)
    ap.add_argument("--repeat_penalty", type=float, default=0.8)
    ap.add_argument("--eos_boost_after", type=int, default=20)
    ap.add_argument("--eos_boost", type=float, default=0.3)
    ap.add_argument("--direction", type=str, default="zh2en",
                    choices=["zh2en", "en2zh", "auto"],
                    help="Direction tag for src: zh2en, en2zh, or auto by language.")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)

    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)

    model = TransformerMT(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=384,
        n_heads=6,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1536,
        dropout=0.1,
        tie_embeddings=True,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    text = args.text.strip()
    if args.direction == "auto":
        if looks_like_chinese(text):
            src_text = "<2en> " + text
        else:
            src_text = "<2zh> " + text
    elif args.direction == "zh2en":
        src_text = "<2en> " + text
    else:
        src_text = "<2zh> " + text

    src_ids = [bos_id] + sp.encode(src_text, out_type=int) + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    out_ids = beam_search(
        model=model,
        sp=sp,
        src_ids=src,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.beams,
        length_penalty=args.len_penalty,
        no_repeat_ngram=args.no_repeat_ngram,
        repeat_penalty=args.repeat_penalty,
        eos_boost_after=args.eos_boost_after,
        eos_boost=args.eos_boost,
    )

    # Debug (optional)
    print("OUT_IDS:", out_ids[:50])
    print("OUT_PIECES:", [sp.id_to_piece(i) for i in out_ids[:30]])

    # strip bos/eos for decode
    if out_ids and out_ids[0] == bos_id:
        out_ids = out_ids[1:]
    if out_ids and out_ids[-1] == eos_id:
        out_ids = out_ids[:-1]

    print(sp.decode(out_ids))


if __name__ == "__main__":
    main()
