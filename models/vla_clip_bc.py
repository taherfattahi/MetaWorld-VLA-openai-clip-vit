import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

class SimpleTextTransformer(nn.Module):
    """
    Lightweight text encoder for your SimpleTokenizer token IDs.
    Padding is assumed to be pad_id=0 (your collector pads with zeros).
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int = 2, n_heads: int = 8, pad_id: int = 0, max_len: int = 64):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)
        B, T = token_ids.shape
        if T > self.max_len:
            token_ids = token_ids[:, : self.max_len]
            T = self.max_len

        pos = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)  # (B,T,D)

        # attention mask: True where tokens are NOT padding for TransformerEncoder src_key_padding_mask
        pad_mask = (token_ids == self.pad_id)  # (B,T) True means "mask out"
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)

        # mean-pool over non-pad tokens
        keep = (~pad_mask).float().unsqueeze(-1)  # (B,T,1)
        x_sum = (x * keep).sum(dim=1)            # (B,D)
        denom = keep.sum(dim=1).clamp(min=1.0)   # (B,1)
        return x_sum / denom


class VLAClipBC(nn.Module):
    """
    Vision-Language-Action BC model:
      - CLIP Vision Transformer (pretrained) for image embedding
      - small Transformer for your text_ids
      - state encoder
      - fusion transformer over 3 tokens [vision, text, state]
      - action head

    Exposes: act(img_t, text_ids, state_t) -> (B, action_dim)
    """
    def __init__(
        self,
        vocab_size: int,
        state_dim: int,
        action_dim: int,
        d_model: int = 512,
        pad_id: int = 0,
        clip_vision_name: str = "openai/clip-vit-base-patch32",
        freeze_vision: bool = True,
        text_layers: int = 2,
        text_heads: int = 8,
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        action_squash: bool = True,  # MetaWorld actions typically in [-1,1]
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.pad_id = pad_id
        self.action_squash = action_squash

        # --- Vision encoder (transformer) ---
        self.vision = CLIPVisionModel.from_pretrained(clip_vision_name)
        vision_hidden = self.vision.config.hidden_size  # e.g. 768

        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False

        self.vision_proj = nn.Linear(vision_hidden, d_model)

        # CLIP normalization constants
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        # --- Text encoder for your SimpleTokenizer ids ---
        self.text = SimpleTextTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=text_layers,
            n_heads=text_heads,
            pad_id=pad_id,
            max_len=64,
        )

        # --- State encoder ---
        self.state_enc = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )

        # Optional normalization for state (filled at train/save time)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

        # --- Fusion over 3 tokens ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=fusion_heads,
            dim_feedforward=4 * d_model,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=fusion_layers)
        self.fusion_ln = nn.LayerNorm(d_model)

        # --- Action head ---
        self.head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, action_dim),
        )

    def _prep_image_for_clip(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B,3,H,W) in [0,1]
        returns: normalized (B,3,224,224)
        """
        if img.dtype != torch.float32:
            img = img.float()
        # Resize to CLIP expected size (224)
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img = (img - self.clip_mean) / self.clip_std
        return img

    def encode_vision(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B,3,H,W) [0,1]
        x = self._prep_image_for_clip(img)
        out = self.vision(pixel_values=x)
        pooled = out.pooler_output  # (B, vision_hidden)
        return self.vision_proj(pooled)  # (B, d_model)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        return self.text(text_ids)  # (B, d_model)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        # normalize state using dataset stats
        s = (state - self.state_mean) / self.state_std.clamp(min=1e-6)
        return self.state_enc(s)  # (B, d_model)

    def forward(self, img: torch.Tensor, text_ids: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        v = self.encode_vision(img)
        t = self.encode_text(text_ids)
        s = self.encode_state(state)

        tokens = torch.stack([v, t, s], dim=1)  # (B,3,D)
        fused = self.fusion(tokens)             # (B,3,D)
        x = self.fusion_ln(fused[:, 0])         # take token 0

        action = self.head(x)                   # (B, action_dim)
        if self.action_squash:
            action = torch.tanh(action)
        return action

    @torch.no_grad()
    def act(self, img_t: torch.Tensor, text_ids: torch.Tensor, state_t: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(img_t, text_ids, state_t)
