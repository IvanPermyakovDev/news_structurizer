from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig

from .config import DEFAULT_MODEL, ModelConfig
from .crf import CRF

class NewsSegmentationModel(nn.Module):
    """
    Token classification model for news segmentation.
    Uses a pretrained transformer encoder with a classification head.
    """

    def __init__(
        self,
        model_name: str = "ai-forever/ruBert-base",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_encoder_layers: int = 0
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze_encoder_layers > 0:
            for layer in self.encoder.encoder.layer[:freeze_encoder_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get boundary predictions."""
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        return predictions


class BiLSTMSegmentationModel(nn.Module):
    """
    Alternative model using BiLSTM on top of transformer embeddings.
    """

    def __init__(
        self,
        model_name: str = "ai-forever/ruBert-base",
        num_labels: int = 2,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> dict:
        with torch.no_grad():
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = encoder_outputs.last_hidden_state

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }


def create_model(
    model_type: str = "transformer",
    model_name: str = "ai-forever/ruBert-base",
    **kwargs
) -> nn.Module:
    if model_type == "transformer":
        return NewsSegmentationModel(model_name=model_name, **kwargs)
    elif model_type == "bilstm":
        return BiLSTMSegmentationModel(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor]


class BoundarySegmenter(nn.Module):
    """
    Encodes chunks (pseudo-sentences) with a pretrained encoder,
    then applies a Transformer over chunk embeddings to predict boundaries.
    """

    def __init__(self, model_cfg: ModelConfig = DEFAULT_MODEL):
        super().__init__()
        self.model_cfg = model_cfg
        self.chunk_encoder = AutoModel.from_pretrained(model_cfg.pretrained_model_name)
        hidden_size = self.chunk_encoder.config.hidden_size
        self.chunk_batch_size = model_cfg.chunk_batch_size

        def _best_heads(embed_dim: int, target: int) -> int:
            divisors = set()
            i = 1
            while i * i <= embed_dim:
                if embed_dim % i == 0:
                    divisors.add(i)
                    divisors.add(embed_dim // i)
                i += 1
            candidates = sorted(d for d in divisors if d >= 1)
            return min(candidates, key=lambda d: (abs(d - target), -d)) if candidates else 1

        preferred_heads = model_cfg.context_heads
        n_heads = preferred_heads if hidden_size % preferred_heads == 0 else _best_heads(hidden_size, preferred_heads)

        self.use_positional = model_cfg.use_positional
        if self.use_positional:
            self.positional = nn.Embedding(model_cfg.positional_max_positions, hidden_size)
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)
        self.cls_dropout = nn.Dropout(model_cfg.classifier_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dropout=model_cfg.context_dropout,
            dim_feedforward=hidden_size * 4,
            batch_first=False,
            activation="gelu",
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_cfg.context_layers)
        self.classifier = nn.Linear(hidden_size, 1)

        # Optional CRF
        self.use_crf = model_cfg.use_crf
        if self.use_crf:
            self.emission_head = nn.Linear(hidden_size, 2)
            self.crf = CRF(2)

        # Optional contrastive loss
        self.use_contrastive = model_cfg.use_contrastive
        self.triplet = (
            nn.TripletMarginLoss(margin=model_cfg.triplet_margin) if self.use_contrastive else None
        )
        self.contrastive_weight = model_cfg.contrastive_weight

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        pos_weight: Optional[float] = None,
    ) -> ForwardOutput:
        batch_size, max_chunks, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention = attention_mask.view(-1, seq_len)
        total_chunks = flat_input_ids.size(0)
        hidden_size = self.chunk_encoder.config.hidden_size

        chunk_embeddings_list = []
        chunk_batch = max(self.chunk_batch_size, 1)
        for start in range(0, total_chunks, chunk_batch):
            end = min(total_chunks, start + chunk_batch)
            encoder_outputs = self.chunk_encoder(
                input_ids=flat_input_ids[start:end],
                attention_mask=flat_attention[start:end],
            )
            last_hidden = encoder_outputs.last_hidden_state
            if self.model_cfg.pooling == "mean":
                mask = flat_attention[start:end].unsqueeze(-1)
                masked = last_hidden * mask
                denom = mask.sum(dim=1).clamp(min=1)
                pooled = masked.sum(dim=1) / denom
            else:
                pooled = last_hidden[:, 0, :]
            chunk_embeddings_list.append(pooled)

        chunk_embeddings = torch.cat(chunk_embeddings_list, dim=0)
        chunk_embeddings = chunk_embeddings.view(batch_size, max_chunks, hidden_size)

        chunk_mask_bool = chunk_mask.bool()
        if self.use_positional:
            max_pos = chunk_embeddings.size(1)
            pos_ids = torch.arange(max_pos, device=chunk_embeddings.device)
            pos_ids = pos_ids.clamp_max(self.model_cfg.positional_max_positions - 1)
            pos_embed = self.positional(pos_ids)
            chunk_embeddings = chunk_embeddings + pos_embed.unsqueeze(0)

        chunk_embeddings = self.pre_ln(chunk_embeddings)
        chunk_embeddings = chunk_embeddings * chunk_mask_bool.unsqueeze(-1)

        ctx_in = chunk_embeddings.transpose(0, 1)
        src_key_padding_mask = ~chunk_mask_bool
        ctx_out = self.context_encoder(ctx_in, src_key_padding_mask=src_key_padding_mask)
        ctx_out = ctx_out.transpose(0, 1)

        logits = self.classifier(self.cls_dropout(ctx_out)).squeeze(-1)

        loss = None
        if labels is not None:
            if self.use_crf:
                emissions = self.emission_head(ctx_out)
                y = labels.long()
                loss = self.crf(emissions, y, mask=chunk_mask_bool, reduction="mean")
            else:
                masked_logits = logits[chunk_mask_bool]
                masked_labels = labels[chunk_mask_bool]
                if masked_logits.numel():
                    if pos_weight is not None:
                        pw_tensor = torch.tensor([pos_weight], dtype=masked_logits.dtype, device=masked_logits.device)
                    else:
                        pw_tensor = None
                    loss = F.binary_cross_entropy_with_logits(masked_logits, masked_labels, pos_weight=pw_tensor)

            if self.use_contrastive and chunk_mask_bool.any():
                contrast = self._contrastive_triplets(ctx_out, labels, chunk_mask_bool)
                if contrast is not None and loss is not None:
                    loss = loss + self.contrastive_weight * contrast
                elif contrast is not None:
                    loss = self.contrastive_weight * contrast

        return ForwardOutput(logits=logits, loss=loss)

    def _contrastive_triplets(
        self, ctx_out: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if self.triplet is None:
            return None
        batch_size, _, _ = ctx_out.shape
        anchors = []
        positives = []
        negatives = []
        for b in range(batch_size):
            valid_len = int(mask[b].sum().item())
            if valid_len < 3:
                continue
            y = labels[b, :valid_len]
            x = ctx_out[b, :valid_len]
            for i in range(1, valid_len - 1):
                if y[i] >= 0.5:
                    anchors.append(x[i])
                    positives.append(x[i - 1])
                    negatives.append(x[i + 1])
        if not anchors:
            return None
        A = torch.stack(anchors)
        P = torch.stack(positives)
        N = torch.stack(negatives)
        return self.triplet(A, P, N)
