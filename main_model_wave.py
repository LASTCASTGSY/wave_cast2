"""
CSDI v2 Main Model for Wave Height Forecasting
===============================================

Summary of Changes vs. V1:
--------------------------
1. ERA5 SPATIAL ENCODER:
   - New SpatialEncoder class using 2D CNN to encode ERA5 grids
   - Processes (B, T_era, C, H, W) → (B, T_era, D_spatial) spatial embeddings
   - Configurable architecture with ResBlocks

2. CROSS-ATTENTION FUSION:
   - ResidualBlock now supports cross-attention to ERA5 context
   - Optional CrossAttention layers for multimodal fusion
   - Follows MCD-TSF paper architecture patterns

3. CLASSIFIER-FREE GUIDANCE (CFG):
   - Support for unconditional training with probability dropout
   - CFG scale parameter for inference
   - Improved generation quality through guidance

4. FORECASTING-SPECIFIC OPTIMIZATIONS:
   - CSDI_Wave_Forecasting_v2 with ERA5 integration
   - Proper handling of context-to-horizon transition
   - Enhanced evaluate() method for multi-sample generation

Author: CSDI Wave Forecasting Project
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

try:
    from linear_attention_transformer import LinearAttentionTransformer
    HAS_LINEAR_ATTN = True
except ImportError:
    HAS_LINEAR_ATTN = False


# ============================================================================
# DIFFUSION EMBEDDINGS AND BASIC LAYERS
# ============================================================================

def get_torch_trans(heads: int = 8, layers: int = 1, channels: int = 64):
    """Standard PyTorch Transformer encoder."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def get_linear_trans(heads: int = 8, layers: int = 1, channels: int = 64):
    """Linear attention transformer for efficiency."""
    if not HAS_LINEAR_ATTN:
        return get_torch_trans(heads, layers, channels)
    return LinearAttentionTransformer(
        dim=channels,
        depth=layers,
        heads=heads,
        max_seq_len=1024,
        n_local_attn_heads=0,
        local_attn_window_size=0,
    )


def Conv1d_with_init(in_channels: int, out_channels: int, kernel_size: int):
    """Conv1d with Kaiming initialization."""
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    """Sinusoidal diffusion step embedding."""
    
    def __init__(self, num_steps: int, embedding_dim: int = 128, 
                 projection_dim: Optional[int] = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: torch.Tensor) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps: int, dim: int = 64) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


# ============================================================================
# ERA5 SPATIAL ENCODER (NEW in V2)
# ============================================================================

class SpatialResBlock(nn.Module):
    """Residual block for 2D spatial encoding."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class SpatialEncoder(nn.Module):
    """
    2D CNN encoder for ERA5 spatial grids.
    
    Encodes (B, T, C, H, W) ERA5 data into (B, T, D) spatial embeddings
    that can be used as conditioning for the diffusion model.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_dim: int = 128,
        num_blocks: int = 2,
        downsample_factor: int = 2
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_dim = out_dim
        
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        
        self.blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        for i in range(num_blocks):
            self.blocks.append(SpatialResBlock(hidden_channels))
            if i < num_blocks - 1 and downsample_factor > 1:
                self.downsample.append(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, stride=2, padding=1)
                )
            else:
                self.downsample.append(nn.Identity())
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ERA5 spatial grids: (B, T, C, H, W) -> (B, T, D)"""
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        
        x = self.input_conv(x)
        x = F.silu(x)
        
        for block, down in zip(self.blocks, self.downsample):
            x = block(x)
            x = down(x)
        
        x = self.global_pool(x)
        x = x.view(B * T, -1)
        x = self.output_proj(x)
        x = x.view(B, T, self.out_dim)
        
        return x


# ============================================================================
# CROSS-ATTENTION MODULE (NEW in V2)
# ============================================================================

class CrossAttention(nn.Module):
    """Cross-attention layer for attending to ERA5 spatial context."""
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Cross-attention: (B, N, D) query, (B, M, D_ctx) context -> (B, N, D)"""
        B, N, _ = x.shape
        M = context.shape[1]
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.view(B, N, self.heads, -1).permute(0, 2, 1, 3)
        k = k.view(B, M, self.heads, -1).permute(0, 2, 1, 3)
        v = v.view(B, M, self.heads, -1).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        
        return self.to_out(out)


# ============================================================================
# RESIDUAL BLOCK WITH OPTIONAL CROSS-ATTENTION
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block for diffusion model with optional cross-attention."""
    
    def __init__(
        self,
        side_dim: int,
        channels: int,
        diffusion_embedding_dim: int,
        nheads: int,
        is_linear: bool = False,
        use_cross_attention: bool = False,
        context_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        self.is_linear = is_linear
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads, layers=1, channels=channels)
        else:
            self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
            self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        
        self.use_cross_attention = use_cross_attention
        if use_cross_attention and context_dim is not None:
            self.cross_attn = CrossAttention(
                query_dim=channels,
                context_dim=context_dim,
                heads=nheads,
                dim_head=channels // nheads
            )
            self.cross_attn_norm = nn.LayerNorm(channels)
        else:
            self.cross_attn = None

    def forward_time(self, y: torch.Tensor, base_shape: Tuple) -> torch.Tensor:
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        
        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y: torch.Tensor, base_shape: Tuple) -> torch.Tensor:
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(
        self,
        x: torch.Tensor,
        cond_info: torch.Tensor,
        diffusion_emb: torch.Tensor,
        era5_context: Optional[torch.Tensor] = None,
        era5_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional ERA5 cross-attention."""
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)
        
        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb
        
        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)
        
        if self.use_cross_attention and self.cross_attn is not None and era5_context is not None:
            y_attn = y.permute(0, 2, 1)
            y_attn = self.cross_attn_norm(y_attn)
            y_attn = y_attn + self.cross_attn(y_attn, era5_context, era5_mask)
            y = y_attn.permute(0, 2, 1)
        
        y = self.mid_projection(y)
        
        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)
        y = y + cond_info
        
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        
        return (x + residual) / math.sqrt(2.0), skip


# ============================================================================
# DIFFUSION MODEL BACKBONE
# ============================================================================

class diff_CSDI(nn.Module):
    """CSDI diffusion backbone with optional ERA5 cross-attention."""
    
    def __init__(
        self,
        config: Dict,
        inputdim: int = 2,
        use_era5: bool = False,
        era5_dim: Optional[int] = None
    ):
        super().__init__()
        self.channels = config["channels"]
        self.use_era5 = use_era5
        
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"],
                is_linear=config.get("is_linear", False),
                use_cross_attention=use_era5,
                context_dim=era5_dim if use_era5 else None
            )
            for _ in range(config["layers"])
        ])

    def forward(
        self,
        x: torch.Tensor,
        cond_info: torch.Tensor,
        diffusion_step: torch.Tensor,
        era5_context: Optional[torch.Tensor] = None,
        era5_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        B, inputdim, K, L = x.shape
        
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(
                x, cond_info, diffusion_emb,
                era5_context=era5_context,
                era5_mask=era5_mask
            )
            skip.append(skip_connection)
        
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        x = x.reshape(B, K, L)
        
        return x


# ============================================================================
# CSDI BASE MODEL
# ============================================================================

class CSDI_base(nn.Module):
    """Base CSDI model for time series imputation/forecasting."""
    
    def __init__(
        self,
        target_dim: int,
        config: Dict,
        device: str,
        use_era5: bool = False,
        era5_config: Optional[Dict] = None
    ):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.use_era5 = use_era5
        
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1
        
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim,
            embedding_dim=self.emb_feature_dim
        )
        
        self.era5_dim = None
        if use_era5 and era5_config is not None:
            self.era5_encoder = SpatialEncoder(
                in_channels=era5_config.get("in_channels", 3),
                hidden_channels=era5_config.get("hidden_channels", 64),
                out_dim=era5_config.get("out_dim", 128),
                num_blocks=era5_config.get("num_blocks", 2)
            )
            self.era5_dim = era5_config.get("out_dim", 128)
        else:
            self.era5_encoder = None
        
        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim
        
        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(
            config_diff, input_dim,
            use_era5=use_era5,
            era5_dim=self.era5_dim
        )
        
        # Diffusion schedule
        self.num_steps = config_diff["num_steps"]
        
        if config_diff["schedule"] == "quad":
            self.betas = np.linspace(
                config_diff["beta_start"] ** 0.5,
                config_diff["beta_end"] ** 0.5,
                self.num_steps
            ) ** 2
        else:
            self.betas = np.linspace(
                config_diff["beta_start"],
                config_diff["beta_end"],
                self.num_steps
            )
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        self.betas = torch.tensor(self.betas, dtype=torch.float32).to(device)
        self.alphas = torch.tensor(self.alphas, dtype=torch.float32).to(device)
        self.alphas_cumprod = torch.tensor(self.alphas_cumprod, dtype=torch.float32).to(device)
        self.alphas_cumprod_prev = torch.tensor(self.alphas_cumprod_prev, dtype=torch.float32).to(device)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.cfg_dropout_prob = config["model"].get("cfg_dropout_prob", 0.0)
        self.cfg_scale = config["model"].get("cfg_scale", 1.0)
        
        # Conditioning schedule
        cond_config = config.get("conditioning", {})
        self.cond_scale_type = cond_config.get("type", "none")  # "exp", "linear", "none"
        self.cond_alpha = cond_config.get("alpha", 1.0)
        self.cond_beta = cond_config.get("beta", 0.1)
        self.cond_min = cond_config.get("min", 0.2)
        
        # Pre-compute conditioning scales for all diffusion steps
        if self.cond_scale_type == "exp":
            self.cond_scales = self.cond_alpha * np.exp(-self.cond_beta * np.arange(self.num_steps))
            self.cond_scales = np.maximum(self.cond_scales, self.cond_min)
        elif self.cond_scale_type == "linear":
            self.cond_scales = np.linspace(1.0, self.cond_min, self.num_steps)
        else:
            self.cond_scales = np.ones(self.num_steps)
        
        self.cond_scales = torch.tensor(self.cond_scales, dtype=torch.float32).to(device)
        
        # Loss weighting
        loss_config = config.get("loss", {})
        self.use_loss_weights = loss_config.get("use_weights", False)
        if self.use_loss_weights:
            loss_weight_map = loss_config.get("weight_map", {})
            # Default: WVHT weight = 5.0, others = 1.0
            self.loss_weights = torch.ones(target_dim, device=device)
            # Import here to avoid circular dependency
            try:
                from dataset_wave import FEATURE_NAMES
            except ImportError:
                # Fallback if import fails
                FEATURE_NAMES = ['WDIR', 'WSPD', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'DEWP']
            for var_name, weight in loss_weight_map.items():
                if var_name in FEATURE_NAMES:
                    var_idx = FEATURE_NAMES.index(var_name)
                    if var_idx < target_dim:
                        self.loss_weights[var_idx] = weight
            print(f"Loss weights: {dict(zip(FEATURE_NAMES[:target_dim], self.loss_weights.cpu().tolist()))}")
        else:
            self.loss_weights = None

    def time_embedding(self, pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        """Sinusoidal time embedding."""
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        """Generate random mask for training."""
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(
        self,
        observed_mask: torch.Tensor,
        for_pattern_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate historical pattern mask."""
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)
        
        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        
        return cond_mask

    def get_side_info(
        self,
        observed_tp: torch.Tensor,
        cond_mask: torch.Tensor,
        diffusion_step: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Construct side information tensor with optional conditioning scale."""
        B, K, L = cond_mask.shape
        
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)
        
        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)
            side_info = torch.cat([side_info, side_mask], dim=1)
        
        # Apply conditioning scale if diffusion step provided
        if diffusion_step is not None and self.cond_scale_type != "none":
            # diffusion_step is shape [B], get corresponding scale
            scale = self.cond_scales[diffusion_step]  # [B]
            scale = scale.view(B, 1, 1, 1)  # Broadcast to side_info shape
            side_info = side_info * scale
        
        return side_info

    def encode_era5(
        self,
        era5_data: torch.Tensor,
        era5_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode ERA5 spatial data."""
        if self.era5_encoder is None or era5_data is None:
            return None, None
        
        era5_emb = self.era5_encoder(era5_data)
        return era5_emb, era5_mask

    def calc_loss(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        side_info: Optional[torch.Tensor],
        is_train: int,
        set_t: int = -1,
        era5_context: Optional[torch.Tensor] = None,
        era5_mask: Optional[torch.Tensor] = None,
        observed_tp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate diffusion loss."""
        B, K, L = observed_data.shape
        
        if is_train != 1:
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        
        # Get side_info with conditioning scale applied if needed
        if side_info is None and observed_tp is not None:
            side_info = self.get_side_info(observed_tp, cond_mask, diffusion_step=t)
        elif side_info is not None and self.cond_scale_type != "none":
            # Apply conditioning scale to existing side_info
            scale = self.cond_scales[t]  # [B]
            scale = scale.view(B, 1, 1, 1)
            side_info = side_info * scale
        
        current_alpha_bar = self.alphas_cumprod[t].reshape(B, 1, 1)
        noise = torch.randn_like(observed_data)
        
        noisy_data = (current_alpha_bar ** 0.5) * observed_data + \
                     (1.0 - current_alpha_bar) ** 0.5 * noise
        
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        
        predicted = self.diffmodel(
            total_input, side_info, t,
            era5_context=era5_context,
            era5_mask=era5_mask
        )
        
        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        
        # Apply per-variable loss weights
        if self.loss_weights is not None:
            # loss_weights shape: [K]
            residual = residual * self.loss_weights.view(1, K, 1)
        
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
        return loss

    def calc_loss_valid(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        observed_mask: torch.Tensor,
        side_info: Optional[torch.Tensor],
        is_train: int,
        era5_context: Optional[torch.Tensor] = None,
        era5_mask: Optional[torch.Tensor] = None,
        observed_tp: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate validation loss over all timesteps."""
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info,
                is_train, set_t=t,
                era5_context=era5_context,
                era5_mask=era5_mask,
                observed_tp=observed_tp
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def set_input_to_diffmodel(
        self,
        noisy_data: torch.Tensor,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor
    ) -> torch.Tensor:
        """Prepare input for diffusion model."""
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)
        return total_input

    def impute(
        self,
        observed_data: torch.Tensor,
        cond_mask: torch.Tensor,
        side_info: torch.Tensor,
        n_samples: int,
        chunk: int = 4,
        era5_context: Optional[torch.Tensor] = None,
        era5_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate samples via reverse diffusion."""
        B, K, L = observed_data.shape
        result = torch.zeros(B, n_samples, K, L, device=self.device)
        
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            bs = end - start
            
            obs_rep = observed_data.unsqueeze(1).expand(-1, bs, -1, -1).reshape(B*bs, K, L)
            mask_rep = cond_mask.unsqueeze(1).expand(-1, bs, -1, -1).reshape(B*bs, K, L)
            side_rep = side_info.unsqueeze(1).expand(-1, bs, -1, -1, -1).reshape(
                B*bs, side_info.shape[1], K, L
            )
            
            if era5_context is not None:
                era5_rep = era5_context.unsqueeze(1).expand(-1, bs, -1, -1).reshape(
                    B*bs, era5_context.shape[1], era5_context.shape[2]
                )
                era5_mask_rep = era5_mask.unsqueeze(1).expand(-1, bs, -1).reshape(
                    B*bs, era5_mask.shape[1]
                ) if era5_mask is not None else None
            else:
                era5_rep = None
                era5_mask_rep = None
            
            x = torch.randn_like(obs_rep)
            
            for t in reversed(range(self.num_steps)):
                beta_t = self.betas[t].view(1, 1, 1)
                alpha_t = self.alphas[t].view(1, 1, 1)
                alpha_bar_t = self.alphas_cumprod[t].view(1, 1, 1)
                
                cond_obs = (mask_rep * obs_rep).unsqueeze(1)
                noisy_target = ((1 - mask_rep) * x).unsqueeze(1)
                diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                
                t_tensor = torch.tensor([t], device=self.device).repeat(B*bs)
                predicted_noise = self.diffmodel(
                    diff_input, side_rep, t_tensor,
                    era5_context=era5_rep,
                    era5_mask=era5_mask_rep
                )
                
                coef1 = 1 / (alpha_t ** 0.5)
                coef2 = beta_t / ((1 - alpha_bar_t) ** 0.5)
                mean = coef1 * (x - coef2 * predicted_noise)
                
                if t > 0:
                    sigma = self.posterior_variance[t].view(1, 1, 1) ** 0.5
                    noise = torch.randn_like(x)
                    x = mean + sigma * noise
                else:
                    x = mean
            
            x = x * (1 - mask_rep) + obs_rep * mask_rep
            result[:, start:end] = x.view(B, bs, K, L)
        
        return result

    def forward(self, batch: Dict, is_train: int = 1) -> torch.Tensor:
        """Forward pass for training/validation."""
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        
        era5_context, era5_mask = None, None
        if self.use_era5 and 'era5_data' in batch:
            era5_data = batch['era5_data'].to(self.device).float()
            era5_mask_raw = batch.get('era5_mask')
            if era5_mask_raw is not None:
                era5_mask_raw = era5_mask_raw.to(self.device).float()
            era5_context, era5_mask = self.encode_era5(era5_data, era5_mask_raw)
        
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)
        
        # side_info will be computed in calc_loss with proper conditioning scale
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        
        return loss_func(
            observed_data, cond_mask, observed_mask, None, is_train,
            era5_context=era5_context, era5_mask=era5_mask, observed_tp=observed_tp
        )

    def evaluate(
        self,
        batch: Dict,
        n_samples: int
    ) -> Tuple[torch.Tensor, ...]:
        """Evaluate model on batch."""
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)
        
        era5_context, era5_mask = None, None
        if self.use_era5 and 'era5_data' in batch:
            era5_data = batch['era5_data'].to(self.device).float()
            era5_mask_raw = batch.get('era5_mask')
            if era5_mask_raw is not None:
                era5_mask_raw = era5_mask_raw.to(self.device).float()
            era5_context, era5_mask = self.encode_era5(era5_data, era5_mask_raw)
        
        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            
            side_info = self.get_side_info(observed_tp, cond_mask)
            
            samples = self.impute(
                observed_data, cond_mask, side_info, n_samples,
                era5_context=era5_context, era5_mask=era5_mask
            )
            
            for i in range(len(cut_length)):
                target_mask[i, ..., 0:cut_length[i].item()] = 0
        
        return samples, observed_data, target_mask, observed_mask, observed_tp

    def process_data(self, batch: Dict):
        """Process batch data. To be implemented by subclasses."""
        raise NotImplementedError


# ============================================================================
# CSDI WAVE IMPUTATION MODEL
# ============================================================================

class CSDI_Wave(CSDI_base):
    """CSDI model for wave height imputation."""
    
    def __init__(
        self,
        config: Dict,
        device: str,
        target_dim: int = 9,
        use_era5: bool = False,
        era5_config: Optional[Dict] = None
    ):
        super().__init__(
            target_dim, config, device,
            use_era5=use_era5, era5_config=era5_config
        )

    def process_data(self, batch: Dict):
        """Process batch data for wave imputation."""
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)
        
        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


# ============================================================================
# CSDI WAVE FORECASTING MODEL (V2)
# ============================================================================

class CSDI_Wave_Forecasting(CSDI_base):
    """CSDI model for wave height forecasting with ERA5 support."""
    
    def __init__(
        self,
        config: Dict,
        device: str,
        target_dim: int = 9,
        use_era5: bool = False,
        era5_config: Optional[Dict] = None
    ):
        super().__init__(
            target_dim, config, device,
            use_era5=use_era5, era5_config=era5_config
        )
        # FIX: Use config value instead of hardcoding "test_pattern"
        # This allows mixed training during training, but still uses gt_mask during eval
        # The parent class already reads config["model"]["target_strategy"], but we override it here
        # So we need to respect the config value to enable mixed training
        self.target_strategy = config["model"].get("target_strategy", "mix")

    def process_data(self, batch: Dict):
        """Process batch data for wave forecasting."""
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask
        
        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )

    def forward(self, batch: Dict, is_train: int = 1) -> torch.Tensor:
        """Forward pass for forecasting with mixed training support."""
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        
        era5_context, era5_mask = None, None
        if self.use_era5 and 'era5_data' in batch:
            era5_data = batch['era5_data'].to(self.device).float()
            era5_mask_raw = batch.get('era5_mask')
            if era5_mask_raw is not None:
                era5_mask_raw = era5_mask_raw.to(self.device).float()
            era5_context, era5_mask = self.encode_era5(era5_data, era5_mask_raw)
        
        # FIX: Enable mixed training during training, but use gt_mask during validation/eval
        if is_train == 0:
            # Validation/eval: use gt_mask (pure forecasting)
            cond_mask = gt_mask
        elif self.target_strategy == "mix":
            # Training with mix: randomly do interpolation OR forecasting
            # This teaches the model smoothness via interpolation tasks
            rand_mask = self.get_randmask(observed_mask)
            cond_mask = observed_mask.clone()
            for i in range(len(cond_mask)):
                mask_choice = np.random.rand()
                if mask_choice > 0.5:
                    # 50% chance: random interpolation task (fills gaps in context/horizon)
                    # This is the same as imputation - teaches smoothness
                    cond_mask[i] = rand_mask[i]
                else:
                    # 50% chance: forecasting task (context only, like gt_mask)
                    # But we still use gt_mask structure to ensure horizon is masked
                    cond_mask[i] = gt_mask[i]
        else:
            # Other strategies: use gt_mask (pure forecasting)
            cond_mask = gt_mask
        
        # side_info will be computed in calc_loss with proper conditioning scale
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        
        return loss_func(
            observed_data, cond_mask, observed_mask, None, is_train,
            era5_context=era5_context, era5_mask=era5_mask, observed_tp=observed_tp
        )


# ============================================================================
# MODEL FACTORY
# ============================================================================

def get_model(
    config: Dict,
    device: str,
    target_dim: int = 9,
    task: str = "imputation",
    use_era5: bool = False,
    era5_config: Optional[Dict] = None
) -> CSDI_base:
    """Factory function to create CSDI model."""
    if task == "imputation":
        return CSDI_Wave(
            config, device, target_dim,
            use_era5=use_era5, era5_config=era5_config
        )
    elif task == "forecasting":
        return CSDI_Wave_Forecasting(
            config, device, target_dim,
            use_era5=use_era5, era5_config=era5_config
        )
    else:
        raise ValueError(f"Unknown task: {task}")
