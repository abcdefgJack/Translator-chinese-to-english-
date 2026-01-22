import math
from dataclasses import dataclass
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TransformerMT(nn.Module):
    """
    Encoder-Decoder Transformer for Machine Translation.
    Masks:
      - src_key_padding_mask: (B, S) where True means PAD (PyTorch uses True=mask)
      - tgt_key_padding_mask: (B, T)
      - tgt causal mask: (T, T) float mask with -inf for blocked positions
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        transformer_kwargs = dict(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # IMPORTANT: inputs are (B,T,dim)
            norm_first=True,
        )
        try:
            self.transformer = nn.Transformer(**transformer_kwargs, enable_nested_tensor=False)
        except TypeError:
            self.transformer = nn.Transformer(**transformer_kwargs)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            # Tie output weights with input embeddings (common in NLP)
            self.lm_head.weight = self.tok_emb.weight

    def _generate_square_subsequent_mask(self, t: int, device: torch.device) -> torch.Tensor:
        """
        PyTorch Transformer expects a causal mask with upper-tri masked.
        Shape: (T, T)
        """
        return torch.triu(torch.ones((t, t), device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        src_ids: torch.Tensor,          # (B, S)
        tgt_in_ids: torch.Tensor,       # (B, T)
        src_pad_mask_4d: torch.Tensor | None = None,  # (B,1,1,S) from your collate
        tgt_pad_mask_4d: torch.Tensor | None = None,  # (B,1,1,T)
        tgt_causal_mask_4d: torch.Tensor | None = None,  # (1,1,T,T)
    ) -> torch.Tensor:
        """
        Returns logits: (B, T, vocab_size)
        Note: We convert your 4D masks into PyTorch nn.Transformer mask formats.
        """

        device = src_ids.device

        # Convert (B,1,1,S) -> (B,S) with True meaning PAD
        if src_pad_mask_4d is not None:
            src_key_padding_mask = ~src_pad_mask_4d.squeeze(1).squeeze(1)  # True for PAD
        else:
            src_key_padding_mask = (src_ids == self.pad_id)

        if tgt_pad_mask_4d is not None:
            tgt_key_padding_mask = ~tgt_pad_mask_4d.squeeze(1).squeeze(1)
        else:
            tgt_key_padding_mask = (tgt_in_ids == self.pad_id)

        # Causal mask: (T,T) float with -inf upper-tri
        t = tgt_in_ids.size(1)
        tgt_mask = self._generate_square_subsequent_mask(t, device)

        # Embeddings
        src = self.dropout(self.pos_emb(self.tok_emb(src_ids) * math.sqrt(self.d_model)))
        tgt = self.dropout(self.pos_emb(self.tok_emb(tgt_in_ids) * math.sqrt(self.d_model)))

        out = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (B,T,d_model)

        logits = self.lm_head(out)  # (B,T,vocab)
        return logits
