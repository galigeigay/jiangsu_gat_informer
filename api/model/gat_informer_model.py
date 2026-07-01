import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import os
import math
import warnings
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .baseclass import BaseModel
from .constants import (
    JIANGSU_CITY2_WIND,
    JIANGSU_CITY2_SOLAR,
    JIANGSU_CITY_WIND_CAPACITY,
    JIANGSU_CITY_SOLAR_CAPACITY,
    get_latest_province_capacity,
)
from api.io.logger import setup_logger

logger = setup_logger(__name__)


class HybridBusinessLoss(nn.Module):
    """Hybrid Huber + two-sided power weighting + approximate MAPE loss.

    The loss is computed in raw power space when y_scaler is available so it
    stays aligned with business Accuracy (1 - MAPE).
    """

    def __init__(
        self,
        delta=1000.0,
        low_power_threshold=2500.0,
        high_power_threshold=8000.0,
        low_power_weight=1.5,
        high_power_weight=1.5,
        max_weight=3.0,
        mape_weight=0.3,
        denom_floor=1000.0,
        std_loss_weight=0.0,
        corr_loss_weight=0.0,
    ):
        super().__init__()
        self.delta = delta
        self.low_power_threshold = low_power_threshold
        self.high_power_threshold = high_power_threshold
        self.low_power_weight = low_power_weight
        self.high_power_weight = high_power_weight
        self.max_weight = max_weight
        self.mape_weight = mape_weight
        self.denom_floor = denom_floor
        # Amplitude regularizers (computed in raw power space when y_scaler provided, else normalized).
        # std_loss_weight: penalize pred std shrinkage relative to target (counters amplitude compression).
        # corr_loss_weight: reward linear correlation between pred and target (improves R2).
        self.std_loss_weight = std_loss_weight
        self.corr_loss_weight = corr_loss_weight

    def forward(self, pred, target, y_scaler=None):
        # Use raw power space when y_scaler is available; otherwise fall back to normalized space.
        if y_scaler is not None:
            y_mean, y_std = y_scaler
            raw_target = target * y_std + y_mean
            raw_pred = pred * y_std + y_mean
        else:
            raw_target = target
            raw_pred = pred

        # ---- 1. Two-sided power weighting in raw power space ----
        low_mask = raw_target < self.low_power_threshold
        high_mask = raw_target > self.high_power_threshold
        weight = torch.ones_like(raw_target, dtype=raw_target.dtype)
        weight = torch.where(
            low_mask,
            torch.full_like(weight, self.low_power_weight),
            weight,
        )
        weight = torch.where(
            high_mask,
            torch.full_like(weight, self.high_power_weight),
            weight,
        )
        weight = torch.clamp(weight, min=1.0, max=self.max_weight)

        # ---- 2. Huber loss in raw power space ----
        error = raw_pred - raw_target
        abs_error = torch.abs(error)
        delta_t = torch.tensor(self.delta, device=abs_error.device, dtype=abs_error.dtype)
        quadratic = 0.5 * error ** 2
        linear = self.delta * (abs_error - 0.5 * self.delta)
        huber = torch.where(abs_error <= delta_t, quadratic, linear)
        huber_loss = (weight * huber).mean()

        # ---- 3. Approximate MAPE term, directly aligned with Accuracy ----
        denom = torch.clamp(torch.abs(raw_target), min=self.denom_floor)
        norm_error = (raw_pred - raw_target) / denom
        mape_loss = torch.abs(norm_error).mean()

        total_loss = huber_loss + self.mape_weight * mape_loss

        # ---- 4. Optional amplitude regularizers (small weights; batch-level stats) ----
        if self.std_loss_weight > 0.0:
            pred_std = raw_pred.std(unbiased=False)
            target_std = raw_target.std(unbiased=False).detach()
            std_loss = ((pred_std - target_std) / (target_std + 1e-6)) ** 2
            total_loss = total_loss + self.std_loss_weight * std_loss
        if self.corr_loss_weight > 0.0:
            pred_centered = raw_pred - raw_pred.mean()
            target_centered = raw_target - raw_target.mean()
            cov = (pred_centered * target_centered).mean()
            pred_s = pred_centered.std(unbiased=False)
            target_s = target_centered.std(unbiased=False).detach()
            corr = cov / (pred_s * target_s + 1e-6)
            corr_loss = 1.0 - corr
            total_loss = total_loss + self.corr_loss_weight * corr_loss

        return total_loss


class GATLayer(nn.Module):
    """Graph attention layer with edge weights and DropEdge regularization.

    Input: x (batch, N, D_in), edge_index (2, E), edge_weight (E,)
    Output: out (batch, N, D_out)
    """

    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, drop_edge=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features
        self.drop_edge = drop_edge  # DropEdge ratio, only active during training

        self.W = nn.Linear(in_features, out_features, bias=True)
        self.att_src = nn.Parameter(torch.Tensor(1, num_heads, 1, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, num_heads, 1, self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.layer_norm = nn.LayerNorm(out_features)

        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

        # Cache dense adjacency and edge-weight matrices to avoid rebuilding every forward pass
        # Register as buffers so they follow .to(device)
        self.register_buffer("_adj_cache", None, persistent=False)
        self.register_buffer("_weight_cache", None, persistent=False)
        self._cache_edge_id = None  # edge_index object id used for the cache

    def _build_dense_adj(self, edge_index, edge_weight, N, device):
        """Build dense adjacency and edge-weight matrices, rebuilding only when edge_index changes."""
        eid = id(edge_index)
        if self._cache_edge_id == eid and self._adj_cache is not None:
            return self._adj_cache, self._weight_cache

        adj = torch.zeros(N, N, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        weight_mat = None
        if edge_weight is not None:
            weight_mat = torch.zeros(N, N, device=device)
            weight_mat[edge_index[0], edge_index[1]] = edge_weight

        self._adj_cache = adj
        self._weight_cache = weight_mat
        self._cache_edge_id = eid
        return adj, weight_mat

    def forward(self, x, edge_index, edge_weight=None, return_attention=False):
        batch, N, _ = x.shape

        # Linear projection -> (batch, N, heads, head_dim)
        h = self.W(x).view(batch, N, self.num_heads, self.head_dim)
        h = h.permute(0, 2, 1, 3)  # (batch, heads, N, head_dim)

        # Compute attention scores for all node pairs
        scores_src = (h * self.att_src).sum(dim=-1, keepdim=True)  # (batch, heads, N, 1)
        scores_dst = (h * self.att_dst).sum(dim=-1, keepdim=True)  # (batch, heads, N, 1)
        scores = scores_src + scores_dst.transpose(-2, -1)          # (batch, heads, N, N)
        scores = self.leaky_relu(scores)

        # Fetch dense adjacency and weights from cache; batch/head dimensions rely on broadcasting
        adj, weight_mat = self._build_dense_adj(edge_index, edge_weight, N, x.device)

        # ---- DropEdge regularization: randomly drop inter-city edges while keeping self-loops ----
        if self.training and self.drop_edge > 0:
            # Inter-city mask, excluding self-loops
            inter_mask = (1 - torch.eye(N, device=adj.device)).bool()
            # Randomly keep/drop inter-city edges
            drop_mask = torch.rand_like(adj) > self.drop_edge
            # Always keep self-loops; keep inter-city edges by probability
            final_mask = (~inter_mask) | (inter_mask & drop_mask)
            adj = adj * final_mask.float()
            if weight_mat is not None:
                weight_mat = weight_mat * final_mask.float()

        # Add edge weights to attention scores when available
        if weight_mat is not None:
            scores = scores + weight_mat.unsqueeze(0).unsqueeze(0)  # broadcast (1,1,N,N)

        # Mask non-edge positions to -inf
        scores = scores.masked_fill(adj.unsqueeze(0).unsqueeze(0) == 0, -1e9)
        alpha = torch.softmax(scores, dim=-1)  # (batch, heads, N, N), row=target node, col=source node
        alpha_msg = self.dropout(alpha)

        # Message passing: alpha @ h
        out = torch.matmul(alpha_msg, h)  # (batch, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, N, self.out_features)
        out = self.layer_norm(out)
        if return_attention:
            return out, alpha
        return out

class InformerEncoderLayer(nn.Module):
    """Informer-style encoder layer: multi-head self-attention + FFN + residual norms.

    This implementation uses PyTorch MultiheadAttention. For T=96 the compute
    cost is acceptable; ProbSparse can be substituted if longer sequences are used.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class InformerEncoder(nn.Module):
    """Informer encoder stack, with optional last-layer attention extraction."""

    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, T, d_model)
            return_attention: bool. If True, return last-layer self-attention weights (batch, T, T)
        Returns:
            x: (batch, T, d_model)
            Optional attn_weights: (batch, T, T), last-layer temporal attention
        """
        attn_weights = None
        for i, layer in enumerate(self.layers):
            if return_attention and i == len(self.layers) - 1:
                # Return attention from the last layer
                attn_out, attn_weights = layer.attention(x, x, x, need_weights=True, average_attn_weights=True)
                x = layer.norm1(x + layer.dropout(attn_out))
            else:
                x = layer(x)
        if return_attention:
            return x, attn_weights
        return x

class SpatiotemporalGATInformer(nn.Module):
    """End-to-end GAT spatial encoder + Informer temporal encoder.

    Input: x (batch, T, N, D_node), edge_index (2, E), edge_weight (E,)
    Output: pred (batch,), single-step regression prediction
    """

    def __init__(
        self,
        num_nodes,
        node_feat_dim,
        d_gat=64,
        gat_heads=4,
        d_model=128,
        informer_layers=2,
        informer_heads=4,
        d_ff=256,
        seq_len=96,
        dropout=0.1,
        drop_edge=0.0,
        temporal_noise=0.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len

        self.gat = GATLayer(node_feat_dim, d_gat, num_heads=gat_heads, dropout=dropout, drop_edge=drop_edge)

        self.spatial_proj = nn.Linear(num_nodes * d_gat, d_model)
        self.proj_norm = nn.LayerNorm(d_model)

        # Learnable positional encoding + year embedding
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        # Supports years 2018-2030
        self.year_embed = nn.Embedding(2030 - 2018 + 1, d_model)

        self.temporal_encoder = InformerEncoder(
            d_model=d_model,
            n_heads=informer_heads,
            d_ff=d_ff,
            num_layers=informer_layers,
            dropout=dropout,
        )

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.temporal_noise = temporal_noise
        self.dropout = nn.Dropout(dropout)

    def _encode_temporal_input(self, x, edge_index, edge_weight=None, year_ids=None):
        """Apply GAT spatial encoding plus position/year encoding for the Informer input."""
        batch, T, N, D = x.shape
        x_flat = x.reshape(batch * T, N, D)
        h_flat = self.gat(x_flat, edge_index, edge_weight)
        x_spatial = h_flat.view(batch, T, N, -1)
        x_temp = x_spatial.reshape(batch, T, -1)
        x_temp = self.spatial_proj(x_temp)
        x_temp = self.proj_norm(x_temp)
        x_temp = x_temp + self.pos_enc[:, :T, :]
        if year_ids is not None:
            year_emb = self.year_embed(year_ids).unsqueeze(1)
            x_temp = x_temp + year_emb
        if self.training and self.temporal_noise > 0:
            noise = torch.randn_like(x_temp) * self.temporal_noise
            x_temp = x_temp + noise
        return x_temp

    def forward(self, x, edge_index, edge_weight=None, year_ids=None):
        # 1-4. GAT spatial encoding + position/year encoding + temporal noise
        x_temp = self._encode_temporal_input(x, edge_index, edge_weight, year_ids)

        # 5. Informer temporal encoding
        x_enc = self.temporal_encoder(x_temp)      # (batch, T, d_model)

        # 6. Predict from the last time step
        x_pool = x_enc[:, -1, :]                   # (batch, d_model)
        out = self.pred_head(x_pool).squeeze(-1)   # (batch,)
        return out

    @torch.no_grad()
    def extract_gat_attention(self, x, edge_index, edge_weight=None):
        """Extract GAT spatial attention weights alpha for visualization.

        Args:
            x: Tensor, shape (batch, T, N, D_node) or (batch, N, D_node)
            edge_index, edge_weight: graph structure

        Returns:
            alpha: np.ndarray, shape:
                - 4D input: (batch, T, heads, N, N)
                - 3D input: (batch, heads, N, N)
              alpha[..., i, j] is the attention weight from node j to node i.
        """
        self.eval()
        # Handle 3D input (batch, N, D) — single time step
        if x.dim() == 3:
            _, alpha = self.gat(x, edge_index, edge_weight, return_attention=True)
            return alpha.cpu().numpy()  # (batch, heads, N, N)

        # Handle 4D input (batch, T, N, D)
        batch, T, N, D = x.shape
        x_flat = x.reshape(batch * T, N, D)
        _, alpha_flat = self.gat(x_flat, edge_index, edge_weight, return_attention=True)
        # alpha_flat: (batch*T, heads, N, N)
        alpha = alpha_flat.view(batch, T, self.gat.num_heads, N, N)
        return alpha.cpu().numpy()

    @torch.no_grad()
    def extract_temporal_attention(self, x, edge_index, edge_weight=None, year_ids=None):
        """Extract Informer temporal self-attention weights.

        Args:
            x: Tensor, shape (batch, T, N, D_node)
            edge_index, edge_weight: graph structure
            year_ids: Tensor or None, shape (batch,)

        Returns:
            attn: np.ndarray, shape (batch, T, T), last-layer temporal attention
        """
        self.eval()
        x_temp = self._encode_temporal_input(x, edge_index, edge_weight, year_ids)
        _, attn = self.temporal_encoder(x_temp, return_attention=True)
        return attn.cpu().numpy()


class GATInformerModel(BaseModel):

    def __init__(
        self,
        label_type="wind",
        seq_len=96,
        d_gat=32,
        gat_heads=4,
        d_model=64,
        informer_layers=2,
        informer_heads=4,
        d_ff=128,
        epochs=100,
        batch_size=32,
        lr=5e-4,
        dropout=0.3,
        corr_threshold=0.3,
        val_ratio=0.2,
        patience=3,
        device=None,
        rolling_backtest=True,
        rb_n_folds=3,
        rb_val_ratio=0.15,
        loss_delta=1.0,
        low_power_threshold=2500.0,
        high_power_threshold=8000.0,
        low_power_weight=1.5,
        high_power_weight=1.5,
        max_sample_weight=3.0,
        mape_weight=0.3,
        drop_edge=0.3,
        temporal_noise=0.05,
        weight_decay=1e-2,
        clip_grad=1.0,
        std_loss_weight=0.0,
        corr_loss_weight=0.0,
        recent_weight=1.0,
        recent_tail_ratio=0.30,
        early_stop_metric="val_loss",
        capacity_clip_ratio=0.98,
    ):
        super().__init__(class_name="GATInformerModel")
        self.label_type = label_type
        self.seq_len = seq_len
        self.d_gat = d_gat
        self.gat_heads = gat_heads
        self.d_model = d_model
        self.informer_layers = informer_layers
        self.informer_heads = informer_heads
        self.d_ff = d_ff
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.corr_threshold = corr_threshold
        self.val_ratio = val_ratio
        self.patience = patience
        self.enable_rolling_backtest = rolling_backtest
        self.rb_n_folds = rb_n_folds
        self.rb_val_ratio = rb_val_ratio
        self.loss_delta = loss_delta
        self.low_power_threshold = low_power_threshold
        self.high_power_threshold = high_power_threshold
        self.low_power_weight = low_power_weight
        self.high_power_weight = high_power_weight
        self.max_sample_weight = max_sample_weight
        self.mape_weight = mape_weight
        self.drop_edge = drop_edge
        self.temporal_noise = temporal_noise
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        # Amplitude regularizers: penalize pred std shrinkage and reward correlation with target.
        self.std_loss_weight = std_loss_weight
        self.corr_loss_weight = corr_loss_weight
        # Recency-weighted sampling: up-weight the most recent training samples to counter
        # concept drift on the latest (Fold3) window. recent_weight is the max multiplicative
        # boost applied to the last recent_tail_ratio fraction of the training set.
        self.recent_weight = recent_weight
        self.recent_tail_ratio = recent_tail_ratio
        # Model-selection metric for early stopping. 'val_loss' (legacy) or 'val_accuracy'.
        self.early_stop_metric = early_stop_metric
        # Capacity cap ratio applied to predictions (upper bound = ratio * province capacity).
        self.capacity_clip_ratio = capacity_clip_ratio

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[GAT+Informer] Using device: {self.device}")

        # Choose city list, capacities, and weather columns by label_type
        if "wind" in label_type:
            self._city_list = list(JIANGSU_CITY2_WIND)
            self._capacity_dict = dict(JIANGSU_CITY_WIND_CAPACITY)
            self._weather_cols = [
                # "windspeed_10m",
                "windspeed_100m",
                "windspeed_200m",
                "pressure_surface",
            ]
        elif "solar" in label_type:
            self._city_list = list(JIANGSU_CITY2_SOLAR)
            self._capacity_dict = dict(JIANGSU_CITY_SOLAR_CAPACITY)
            self._weather_cols = [
                "temperature_2m",
                "apparent_temperature",
                "cloudcover",
                "shortwave_radiation_instant",
            ]
        else:
            self._city_list = list(JIANGSU_CITY2_WIND)
            self._capacity_dict = {}
            self._weather_cols = []

        self._model = None
        self.edge_index = None
        self.edge_weight = None
        self._node_feat_dim = None
        self._x_scaler = None   # (mean, std) for input normalization
        self._y_scaler = None   # (mean, std) for label normalization
        self._global_feat_cols = None  # Feature column order captured during training for prediction consistency
        self.loss_history = None
        # Physical upper bound for predictions: ratio * province installed capacity (MW).
        # None when no capacity is configured -> upper clip skipped, only lower bound [0, inf) applied.
        _prov_cap = get_latest_province_capacity(label_type)
        self._capacity_cap = (
            float(_prov_cap) * self.capacity_clip_ratio
            if (_prov_cap and _prov_cap > 0 and self.capacity_clip_ratio > 0)
            else None
        )
        if self._capacity_cap is not None:
            logger.info(
                f"[GAT+Informer] Prediction capacity cap = {self._capacity_cap:.1f} MW "
                f"({self.capacity_clip_ratio:.2f} * {float(_prov_cap):.1f}) for label_type={label_type}"
            )

    def _get_time_col(self, pdf):

        if "date_time" in pdf.columns:
            return "date_time"
        if "time_idx" in pdf.columns:
            logger.warning(
                "[GAT+Informer] date_time column not found; falling back to time_idx sorting. "
                "If upstream data is not date/time ordered, Informer windows may be invalid."
            )
            return "time_idx"
        return None

    def _build_graph(self, pdf_data):
        """Build city graph from historical weather correlations and capacity coupling.

        Edge weight formula:
            W_ij = ReLU(|corr(feat_i, feat_j)| - threshold) * sqrt(capa_i * capa_j) / max_capa

        1. Symmetry: for every inter-city edge above the threshold add both (i,j) and (j,i).
        2. Missing-data cities are connected to all others with weak symmetric edges.
        3. Self-loops use a small fixed residual weight to keep spatial aggregation from
           being dominated by each city's own capacity.
        """
        main_pattern = (
            "windspeed_100m_"
            if "wind" in self.label_type
            else "shortwave_radiation_instant_"
        )

        city_data = {}
        for city in self._city_list:
            col = f"{main_pattern}{city}"
            if col in pdf_data.columns:
                city_data[city] = pdf_data[col].values

        N = len(self._city_list)

        # Capacities and normalization based on the full city list, not just observed ones
        full_capacities = [self._capacity_dict.get(c, 0.0) for c in self._city_list]
        max_capa = max(full_capacities) if max(full_capacities) > 0 else 1.0

        # Log-capacity normalization coefficients to compress head-city dominance
        log_capacities = [math.log1p(c) for c in full_capacities]
        log_max = math.log1p(max_capa)

        def _self_loop_weight(idx):
            """Self-loop weight: small fixed residual to let spatial attention dominate."""
            return 0.15

        def _log_capa_coupling(idx_i, idx_j):
            if log_max <= 0:
                return 0.0
            return (log_capacities[idx_i] * log_capacities[idx_j]) / (log_max ** 2)

        # Degrade to fully-connected graph when < 2 cities have data
        if len(city_data) < 2:
            logger.warning("Insufficient city data for graph; degrading to fully-connected log-capacity graph")
            edge_index = []
            edge_weights = []
            for i in range(N):
                for j in range(N):
                    edge_index.append([i, j])
                    if i == j:
                        w = _self_loop_weight(i)
                    else:
                        w = _log_capa_coupling(i, j)
                    edge_weights.append(max(float(w), 1e-3))
            self.edge_index = (
                torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
            )
            self.edge_weight = torch.tensor(edge_weights, dtype=torch.float32).to(self.device)
            return

        # Pearson correlation matrix among available cities
        avail_cities = list(city_data.keys())
        n_avail = len(avail_cities)
        corr_matrix = np.zeros((n_avail, n_avail))
        for i, ci in enumerate(avail_cities):
            for j, cj in enumerate(avail_cities):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr = np.corrcoef(city_data[ci], city_data[cj])[0, 1]
                    corr_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0.0

        # Use a set + force symmetry: record both (i,j) and (j,i)
        # key: (src_idx, dst_idx)，val: weight
        edge_dict = {}

        for i, ci in enumerate(avail_cities):
            for j, cj in enumerate(avail_cities):
                if i == j:
                    continue
                corr = corr_matrix[i, j]
                relu_corr = max(corr - self.corr_threshold, 0.0)
                if relu_corr <= 0:
                    continue
                src_idx = self._city_list.index(ci)
                dst_idx = self._city_list.index(cj)
                # Log-capacity coupling to compress Yancheng dominance
                capa_coupling = _log_capa_coupling(src_idx, dst_idx)
                weight = float(relu_corr * capa_coupling)
                if weight <= 0:
                    continue
                # Symmetric add (max avoids precision loss from mismatch)
                edge_dict[(src_idx, dst_idx)] = max(edge_dict.get((src_idx, dst_idx), 0.0), weight)
                edge_dict[(dst_idx, src_idx)] = max(edge_dict.get((dst_idx, src_idx), 0.0), weight)

        # Connect missing-data cities to all others with weak symmetric edges
        # Weak weight = corr_threshold * capa_coupling (reachability without signal dominance)
        avail_set = set(avail_cities)
        for idx in range(N):
            city = self._city_list[idx]
            if city in avail_set:
                continue
            for other_idx in range(N):
                if other_idx == idx:
                    continue
                capa_coupling = _log_capa_coupling(idx, other_idx)
                weak_w = float(self.corr_threshold * capa_coupling)
                if weak_w <= 0:
                    weak_w = 1e-3  # Ensure reachability
                # Do not overwrite stronger existing correlation edges
                if (idx, other_idx) not in edge_dict:
                    edge_dict[(idx, other_idx)] = weak_w
                if (other_idx, idx) not in edge_dict:
                    edge_dict[(other_idx, idx)] = weak_w

        # Self-loops on the same scale as inter-city edges
        for idx in range(N):
            self_w = _self_loop_weight(idx)
            edge_dict[(idx, idx)] = self_w

        # Build tensors
        edges = list(edge_dict.keys())
        weights = [edge_dict[e] for e in edges]
        edge_index = [[e[0] for e in edges], [e[1] for e in edges]]

        self.edge_index = (
            torch.tensor(edge_index, dtype=torch.long).contiguous().to(self.device)
        )
        self.edge_weight = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # Validate symmetry
        edge_set = set(edge_dict.keys())
        n_asym = sum(1 for (i, j) in edge_set if i != j and (j, i) not in edge_set)
        if n_asym > 0:
            logger.warning(f"[GAT] Detected {n_asym} asymmetric edges, check _build_graph logic")

        n_self = sum(1 for (i, j) in edges if i == j)
        n_inter = len(edges) - n_self
        logger.info(
            f"[GAT] Graph built: {N} nodes, {len(edges)} edges (self-loops {n_self} + inter-city {n_inter}), "
            f"avg inter-city weight={(sum(w for (i, j), w in edge_dict.items() if i != j) / max(n_inter, 1)):.4f}, "
            f"avg self-loop weight={(sum(w for (i, j), w in edge_dict.items() if i == j) / max(n_self, 1)):.4f}, "
            f"threshold={self.corr_threshold}"
        )

    def _tabular_to_spatiotemporal(self, pdf_data, label_col=None, return_sort_order=False, return_year_ids=False):
        """Convert tabular DataFrame to spatiotemporal tensor (B, T, N, D).

        Args:
            pdf_data: pd.DataFrame with city weather features and optional label
            label_col: str label column name; None to skip y
            return_sort_order: bool, if True also returns original indices to remap predictions
            return_year_ids: bool, if True also returns year IDs for year embeddings

        Returns:
            X: np.ndarray, shape (B, T, N, D_node)
            y: np.ndarray or None, shape (B,)
            Optional sort_idx: np.ndarray, shape (T_total,)
            Optional year_ids: np.ndarray, shape (B,), year index 2018->0 ... 2030->12
        """
        pdf = pdf_data.copy()
        # Preserve original indices so predictions can be mapped back to original DataFrame order
        pdf["__orig_idx__"] = np.arange(len(pdf), dtype=np.int64)

        time_col = self._get_time_col(pdf)
        if time_col:
            pdf = pdf.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        else:
            pdf = pdf.reset_index(drop=True)

        sort_idx = pdf["__orig_idx__"].values.copy()
        pdf = pdf.drop(columns=["__orig_idx__"])

        T_total = len(pdf)
        if T_total < self.seq_len:
            raise ValueError(
                f"Data length {T_total} is smaller than seq_len {self.seq_len}"
            )

        city_list = self._city_list
        weather_cols = self._weather_cols
        N = len(city_list)

        # Build node features: assign ALL city-specific numeric columns to the corresponding city node.
        # Columns containing exactly one city name are node features; columns with 0 or >=2 city names are global.
        # This ensures lag/diff/mean city-specific features (e.g., windspeed_100m_盐城市_lag_4) are node data,
        # preventing them from being broadcast globally and inflating the global feature dimension.
        numeric_cols = list(pdf.select_dtypes(include=[np.number]).columns)
        excluded_patterns = ["time_idx", "date_time", "version", "meteo_source"]
        city_specific_cols = {city: [] for city in city_list}

        for col in numeric_cols:
            if col == label_col:
                continue
            skip = False
            for pat in excluded_patterns:
                if pat in col:
                    skip = True
                    break
            if skip:
                continue
            if col.startswith("capacity_wind_") or col.startswith("capacity_solar_"):
                continue
            if "_capa_w_" in col or "_x_capa_" in col:
                continue
            if col.endswith("_wind_weighted") or col.endswith("_solar_weighted"):
                continue

            matched_cities = []
            # Match both full city names (盐城市) and short aliases used by engineered columns (盐城).
            for c in city_list:
                short_name = c[:-1] if c.endswith("市") else c
                if c in col or short_name in col:
                    matched_cities.append(c)

            # Regional mean features are node-level regional context, not global features broadcast to all cities.
            if "_southern_mean" in col:
                southern_cities = ['南通市', '常州市', '无锡市', '苏州市']
                matched_cities.extend([c for c in southern_cities if c in city_list])
            if "_northern_mean" in col:
                northern_cities = ['盐城市', '连云港市', '宿迁市', '徐州市', '淮安市']
                matched_cities.extend([c for c in northern_cities if c in city_list])

            for city in sorted(set(matched_cities)):
                city_specific_cols[city].append(col)

        # Unify column set so every city has the same feature dimension (zero-pad missing ones)
        all_city_specific = sorted(set().union(*city_specific_cols.values()))
        if all_city_specific:
            node_feats = []
            for city in city_list:
                city_feat = []
                for col in all_city_specific:
                    if col in city_specific_cols[city]:
                        vals = pdf[col].values.astype(np.float32)
                        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                        city_feat.append(vals)
                    else:
                        city_feat.append(np.zeros(T_total, dtype=np.float32))
                node_feats.append(np.stack(city_feat, axis=1))
            node_data = np.stack(node_feats, axis=1)
        else:
            node_data = np.zeros((T_total, N, 1), dtype=np.float32)

        # All city-specific columns must be excluded from global broadcasts
        excluded_cols = set(all_city_specific)

        global_feat_cols = []
        for col in numeric_cols:
            if col == label_col:
                continue
            if col in excluded_cols:
                continue
            if "_capa_w_" in col:
                continue
            if "_x_capa_" in col:
                continue
            if col.startswith("capacity_wind_") or col.startswith("capacity_solar_"):
                continue
            if col.endswith("_wind_weighted") or col.endswith("_solar_weighted"):
                continue
            global_feat_cols.append(col)

        # Determine final global feature columns and extract/concatenate consistently
        if label_col is not None:
            # Training: record global feature column order
            self._global_feat_cols = global_feat_cols.copy() if global_feat_cols else None
            if self._global_feat_cols:
                logger.info(
                    f"[GAT+Informer] Global features {len(global_feat_cols)} dims: {global_feat_cols[:5]}..."
                )
        else:
            # Prediction: reuse recorded column order to keep dimensions consistent
            if self._global_feat_cols is not None:
                global_feat_cols = self._global_feat_cols.copy()

        if global_feat_cols:
            global_data = []
            for col in global_feat_cols:
                if col in pdf.columns:
                    vals = pdf[col].values.astype(np.float32)
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    global_data.append(vals)
                else:
                    global_data.append(np.zeros(T_total, dtype=np.float32))
            global_data = np.stack(global_data, axis=1)
            global_data = np.repeat(global_data[:, None, :], N, axis=1)
            node_data = np.concatenate([node_data, global_data], axis=-1)

        # Build samples with sliding windows
        X, y, year_ids = [], [], []
        for i in range(T_total - self.seq_len + 1):
            X.append(node_data[i : i + self.seq_len])
            if label_col is not None and label_col in pdf.columns:
                y.append(pdf[label_col].iloc[i + self.seq_len - 1])
            # Year of the last time step in the window (for year embedding)
            if return_year_ids:
                ts_val = pdf[time_col].iloc[i + self.seq_len - 1]
                # Support datetime or string
                if isinstance(ts_val, str):
                    ts_val = pd.to_datetime(ts_val)
                year = ts_val.year if hasattr(ts_val, 'year') else 2024
                # Clamp to 2018-2030 range
                year_idx = max(0, min(year - 2018, 12))
                year_ids.append(year_idx)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32) if y else None
        year_ids = np.array(year_ids, dtype=np.int64) if year_ids else None

        if y is not None:
            valid_mask = np.isfinite(y)
            if not valid_mask.all():
                n_invalid = (~valid_mask).sum()
                logger.warning(f"Filtered {n_invalid} invalid label samples")
                X = X[valid_mask]
                y = y[valid_mask]
                if year_ids is not None:
                    year_ids = year_ids[valid_mask]

        if return_sort_order and return_year_ids:
            return X, y, sort_idx, year_ids
        if return_sort_order:
            return X, y, sort_idx
        if return_year_ids:
            return X, y, year_ids
        return X, y

    def train(self, **kwargs):

        feature = label = None
        for k, v in kwargs.items():
            if "data" in k or "feat" in k or "feature" in k:
                feature = v
            if "label" in k or "target" in k:
                label = v

        if feature is None:
            raise ValueError("train() requires a feature argument")

        feature = feature.reset_index(drop=True)
        if label is not None:
            label = label.reset_index(drop=True)
            pdf_all = pd.concat([feature, label], axis=1)
            label_col = label.columns[0]
        else:
            pdf_all = feature.copy()
            label_col = None

        # Build graph structure (only on first training)
        if self.edge_index is None:
            self._build_graph(pdf_all)

        # Data conversion (also extracts year IDs to mitigate cross-year drift)
        X, y, year_ids = self._tabular_to_spatiotemporal(pdf_all, label_col, return_year_ids=True)
        if len(X) == 0:
            raise RuntimeError("No valid samples after sequence construction; check that data length is greater than seq_len")

        self._node_feat_dim = X.shape[3]
        n_samples = len(X)
        n_val = int(n_samples * self.val_ratio)
        n_train = n_samples - n_val

        # Temporal split: last val_ratio as validation
        X_train, y_train, yids_train = X[:n_train], y[:n_train], year_ids[:n_train]
        X_val, y_val, yids_val = X[n_train:], y[n_train:], year_ids[n_train:]

        # Compute train set mean/std for normalization
        x_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)   # (1, 1, 1, D)
        x_std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6
        self._x_scaler = (x_mean, x_std)
        if y is not None:
            y_mean = float(y_train.mean())
            y_std = float(y_train.std()) + 1e-6
            self._y_scaler = (y_mean, y_std)
        else:
            self._y_scaler = (0.0, 1.0)

        # Snapshot training label distribution in RAW (pre-normalization) space so that
        # the label drift check in predict() can compare against denormalized predictions
        # on the same scale (previously this was captured AFTER normalization, making
        # `train_mean=0` always mismatch the denormalized `pred_mean`).
        self._train_label_dist = {
            "mean": float(np.mean(y_train)),
            "std": float(np.std(y_train)),
            "min": float(np.min(y_train)),
            "max": float(np.max(y_train)),
            "p05": float(np.percentile(y_train, 5)),
            "p95": float(np.percentile(y_train, 95)),
        }

        # Normalize
        X_train = (X_train - x_mean) / x_std
        X_val = (X_val - x_mean) / x_std
        y_train = (y_train - self._y_scaler[0]) / self._y_scaler[1]
        y_val = (y_val - self._y_scaler[0]) / self._y_scaler[1]

        logger.info(
            f"[GAT+Informer] Train samples: {n_train}, Val samples: {n_val}, "
            f"seq_len: {self.seq_len}, num_nodes: {len(self._city_list)}, "
            f"node_feat_dim: {self._node_feat_dim}, "
            f"label scaling (mean={self._y_scaler[0]:.2f}, std={self._y_scaler[1]:.2f})"
        )
        # Feature distribution snapshot is captured in normalized space (consistent with
        # the feature drift check in predict()).
        self._train_feat_dist = {
            "mean": float(X_train.mean()),
            "std": float(X_train.std()),
            "min": float(X_train.min()),
            "max": float(X_train.max()),
        }
        logger.info(
            f"[GAT+Informer] Train label distribution (raw scale): "
            f"mean={self._train_label_dist['mean']:.1f}, std={self._train_label_dist['std']:.1f}, "
            f"min={self._train_label_dist['min']:.1f}, max={self._train_label_dist['max']:.1f}, "
            f"p5/p95=({self._train_label_dist['p05']:.1f}, {self._train_label_dist['p95']:.1f})"
        )
        logger.info(
            f"[GAT+Informer] Train feature distribution (normalized): "
            f"mean={self._train_feat_dist['mean']:.4f}, std={self._train_feat_dist['std']:.4f}"
        )

        # Initialize network with regularization params
        self._model = SpatiotemporalGATInformer(
            num_nodes=len(self._city_list),
            node_feat_dim=self._node_feat_dim,
            d_gat=self.d_gat,
            gat_heads=self.gat_heads,
            d_model=self.d_model,
            informer_layers=self.informer_layers,
            informer_heads=self.informer_heads,
            d_ff=self.d_ff,
            seq_len=self.seq_len,
            dropout=self.dropout,
            drop_edge=self.drop_edge,
            temporal_noise=self.temporal_noise,
        ).to(self.device)

        # DataLoader with year_ids
        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train),
            torch.from_numpy(yids_train)
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val),
            torch.from_numpy(yids_val)
        )
        # Use pin_memory and multiple workers to speed up CPU->GPU transfer
        loader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=(self.device == "cuda" or
                        (isinstance(self.device, torch.device) and self.device.type == "cuda")),
            persistent_workers=True,
        )
        # Recency-weighted sampling: up-weight the most recent training samples to counter
        # concept drift on the latest (Fold3) window. Train set is temporally ordered
        # (X_train = X[:n_train]), so a larger index = more recent. We linearly ramp the
        # sample weight from 1.0 over the head (1 - recent_tail_ratio) up to recent_weight
        # at the tail. Uses WeightedRandomSampler (replacement=True) instead of shuffle.
        train_sampler = None
        if self.recent_weight > 1.0 + 1e-6 and 0.0 < self.recent_tail_ratio < 1.0:
            n_train = len(X_train)
            positions = np.arange(n_train, dtype=np.float64) / max(n_train - 1, 1)
            tail_start = 1.0 - self.recent_tail_ratio
            ramp = np.clip((positions - tail_start) / max(self.recent_tail_ratio, 1e-6), 0.0, 1.0)
            sample_weights = 1.0 + (self.recent_weight - 1.0) * ramp
            sample_weights = sample_weights / sample_weights.mean()  # keep mean weight ~1
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=n_train,
                replacement=True,
            )
            logger.info(
                f"[GAT+Informer] Recency sampler: recent_weight={self.recent_weight}, "
                f"tail_ratio={self.recent_tail_ratio}, head_w=1.0, tail_w={float(sample_weights[-1]):.3f} "
                f"(mean-normalized)"
            )
            train_loader = DataLoader(train_ds, sampler=train_sampler, drop_last=False, **loader_kwargs)
        else:
            train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

        # Optimizer with ReduceLROnPlateau: start from initial lr and decay only when
        # val_loss plateaus, avoiding the warmup "rise-then-fall" behavior.
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        min_lr = self.lr * 0.01  # floor at 1% of initial LR
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            threshold=1e-7,
            min_lr=min_lr,
        )
        # Create a normalized-space criterion for training so loss values are ~1-10
        # instead of ~10^6 MW-scale. The thresholds and delta are scaled by y_std so
        # the normalized loss is equivalent to the raw-space loss up to a constant factor.
        y_std = self._y_scaler[1]
        if y_std > 1e-6:
            criterion = HybridBusinessLoss(
                delta=self.loss_delta / y_std,
                low_power_threshold=self.low_power_threshold / y_std,
                high_power_threshold=self.high_power_threshold / y_std,
                low_power_weight=self.low_power_weight,
                high_power_weight=self.high_power_weight,
                max_weight=self.max_sample_weight,
                mape_weight=self.mape_weight,
                denom_floor=1000.0 / y_std,
                std_loss_weight=self.std_loss_weight,
                corr_loss_weight=self.corr_loss_weight,
            )
            logger.info(
                f"[GAT+Informer] Normalized-space loss: delta={self.loss_delta / y_std:.4f}, "
                f"thresholds=({self.low_power_threshold / y_std:.4f}, {self.high_power_threshold / y_std:.4f}), "
                f"denom_floor={1000.0 / y_std:.4f} (y_std={y_std:.2f})"
            )
        else:
            criterion = HybridBusinessLoss(
                delta=self.loss_delta,
                low_power_threshold=self.low_power_threshold,
                high_power_threshold=self.high_power_threshold,
                low_power_weight=self.low_power_weight,
                high_power_weight=self.high_power_weight,
                max_weight=self.max_sample_weight,
                mape_weight=self.mape_weight,
                std_loss_weight=self.std_loss_weight,
                corr_loss_weight=self.corr_loss_weight,
            )
            logger.info(
                "[GAT+Informer] Using HybridBusinessLoss: delta={}, "
                "low_power_weight={}, high_power_weight={}, max_weight={}, mape_weight={}, "
                "thresholds=({}, {})".format(
                    self.loss_delta, self.low_power_weight, self.high_power_weight,
                    self.max_sample_weight, self.mape_weight,
                    self.low_power_threshold, self.high_power_threshold,
                )
            )
        logger.info(
            f"[GAT+Informer] Regularization/capacity: d_gat={self.d_gat}, d_model={self.d_model}, "
            f"layers={self.informer_layers}, d_ff={self.d_ff}, dropout={self.dropout}, "
            f"drop_edge={self.drop_edge}, temporal_noise={self.temporal_noise}, "
            f"weight_decay={self.weight_decay}, std_loss_weight={self.std_loss_weight}, "
            f"corr_loss_weight={self.corr_loss_weight}, early_stop_metric={self.early_stop_metric}, "
            f"lr_scheduler=ReduceLROnPlateau(factor=0.5, patience=3)"
        )

        best_val_loss = float("inf")
        best_val_accuracy = -float("inf")
        best_state = None
        patience_counter = 0

        # Mixed precision training (only enabled for CUDA)
        use_amp = False
        # use_amp = self.device == "cuda" or (isinstance(self.device, torch.device) and self.device.type == "cuda")
        scaler = GradScaler() if use_amp else None

        train_loss_history = []
        val_loss_history = []
        val_acc_history = []
        lr_history = []

        for epoch in range(self.epochs):
            self._model.train()
            train_loss = 0.0
            for bx, by, byid in train_loader:
                bx, by, byid = bx.to(self.device), by.to(self.device), byid.to(self.device)
                optimizer.zero_grad()
                if use_amp:
                    with autocast():
                        pred = self._model(bx, self.edge_index, self.edge_weight, year_ids=byid)
                        loss = criterion(pred, by)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = self._model(bx, self.edge_index, self.edge_weight, year_ids=byid)
                    loss = criterion(pred, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clip_grad)
                    optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= len(train_ds)

            self._model.eval()
            val_loss = 0.0
            val_preds_all = []
            val_y_all = []
            with torch.no_grad():
                for bx, by, byid in val_loader:
                    bx, by, byid = bx.to(self.device), by.to(self.device), byid.to(self.device)
                    if use_amp:
                        with autocast():
                            pred = self._model(bx, self.edge_index, self.edge_weight, year_ids=byid)
                            loss = criterion(pred, by)
                    else:
                        pred = self._model(bx, self.edge_index, self.edge_weight, year_ids=byid)
                        loss = criterion(pred, by)
                    val_loss += loss.item() * len(bx)
                    val_preds_all.append(pred.cpu().numpy())
                    val_y_all.append(by.cpu().numpy())
            val_loss /= len(val_ds)

            # Compute R2 in normalized space (R2 is insensitive to linear scaling)
            val_preds_all = np.concatenate(val_preds_all)
            val_y_all = np.concatenate(val_y_all)
            ss_res = np.sum((val_y_all - val_preds_all) ** 2)
            ss_tot = np.sum((val_y_all - np.mean(val_y_all)) ** 2)
            val_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

            # Compute validation MAPE in raw power space (denormalized)
            if self._y_scaler is not None:
                y_mean, y_std = self._y_scaler
                val_preds_raw = val_preds_all * y_std + y_mean
                val_y_raw = val_y_all * y_std + y_mean
                # Avoid division by zero; clamp denominator to 1000 for true < 1000 (same as calculate_accuracy)
                denom = np.maximum(np.abs(val_y_raw), 1000.0)
                mape = np.mean(np.abs(val_preds_raw - val_y_raw) / denom)
            else:
                mape = np.mean(np.abs(val_preds_all - val_y_all) / (np.maximum(np.abs(val_y_all), 1e-6)))
            accuracy = 1.0 - mape

            train_loss_history.append(float(train_loss))
            val_loss_history.append(float(val_loss))
            val_acc_history.append(float(accuracy))
            lr_history.append(float(optimizer.param_groups[0]["lr"]))
            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Amplitude diagnosis: if pred_std << y_std, model is compressing amplitude and avoiding extremes
                pred_std = float(np.std(val_preds_raw)) if self._y_scaler is not None else float(np.std(val_preds_all))
                y_std_raw = float(np.std(val_y_raw)) if self._y_scaler is not None else float(np.std(val_y_all))
                logger.info(
                    f"[GAT+Informer] Epoch {epoch + 1}/{self.epochs}, "
                    f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, "
                    f"Val R2={val_r2:.4f}, Val Accuracy={accuracy:.4f}, "
                    f"Val pred_std={pred_std:.1f}, y_std={y_std_raw:.1f}, "
                    f"lr={lr_history[-1]:.2e}"
                )

            # Early stopping / checkpointing. LR reduction still follows val_loss, but the
            # saved best_state can follow the business metric (val_accuracy) when requested.
            if self.early_stop_metric == "val_accuracy":
                is_better = accuracy > best_val_accuracy + 1e-4
            else:
                is_better = val_loss < best_val_loss - 1e-7

            if is_better:
                best_val_loss = val_loss
                best_val_accuracy = accuracy
                best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        f"[GAT+Informer] Early stopping at Epoch {epoch + 1}, "
                        f"monitor={self.early_stop_metric}, best Val Loss={best_val_loss:.6f}, "
                        f"best Val Accuracy={best_val_accuracy:.4f}"
                    )
                    break

        self.loss_history = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "val_accuracy": val_acc_history,
            "lr": lr_history,
            "best_val_accuracy": float(best_val_accuracy),
            "best_val_loss": float(best_val_loss),
        }
        self._plot_loss_curve()

        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model = self._model.to(self.device)
            logger.info(f"[GAT+Informer] Training complete, best Val Loss={best_val_loss:.6f}")

        if self.enable_rolling_backtest:
            self.rolling_backtest(
                pdf_all,
                label_col,
                n_folds=self.rb_n_folds,
                val_window_ratio=self.rb_val_ratio,
            )

    def _plot_loss_curve(self, save_path=None):
        """Plot train/validation loss curves using English-only labels to avoid font warnings."""
        if not self.loss_history:
            return None
        train_loss = self.loss_history.get("train_loss", [])
        val_loss = self.loss_history.get("val_loss", [])
        lr_history = self.loss_history.get("lr", [])
        if not train_loss or not val_loss:
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_path is None:
            save_dir = os.path.join("local", "checkpoints", "visualizations")
            os.makedirs(save_dir, exist_ok=True)
            safe_label = str(self.label_type).replace(os.sep, "_")
            save_path = os.path.join(save_dir, f"gat_informer_loss_{safe_label}.png")

        epochs = np.arange(1, len(train_loss) + 1)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(epochs, train_loss, label="Train Loss", linewidth=1.8)
        ax1.plot(epochs, val_loss, label="Val Loss", linewidth=1.8)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("GAT-Informer Training Curve")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="upper right")

        if lr_history:
            ax2 = ax1.twinx()
            ax2.plot(epochs, lr_history, label="Learning Rate", color="gray", linestyle="--", alpha=0.7)
            ax2.set_ylabel("Learning Rate")
            ax2.legend(loc="upper center")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[GAT+Informer] Loss curve saved: {save_path}")
        return save_path

    def rolling_backtest(self, pdf_data, label_col, n_folds=None, val_window_ratio=None):
        """Rolling backtest over multiple time windows using the final model.

        This does NOT retrain per fold; it evaluates the final model on different
        historical validation windows to assess temporal stability.
        """
        if self._model is None:
            logger.warning("[Rolling Backtest] Model is not trained; skipping backtest")
            return None

        n_folds = n_folds or self.rb_n_folds
        val_window_ratio = val_window_ratio or self.rb_val_ratio

        X, y, year_ids = self._tabular_to_spatiotemporal(pdf_data, label_col, return_year_ids=True)
        if y is None or len(X) == 0:
            logger.warning("[Rolling Backtest] No valid samples; skipping backtest")
            return None

        time_col = self._get_time_col(pdf_data)
        pdf_sorted = pdf_data.copy()
        if time_col:
            pdf_sorted = pdf_sorted.sort_values(time_col, kind="mergesort").reset_index(drop=True)
        else:
            pdf_sorted = pdf_sorted.reset_index(drop=True)

        all_target_rows = np.arange(len(pdf_sorted) - self.seq_len + 1, dtype=np.int64) + self.seq_len - 1
        target_rows = all_target_rows
        if label_col in pdf_sorted.columns and len(all_target_rows) > 0:
            target_values = pdf_sorted[label_col].iloc[all_target_rows].values.astype(np.float32)
            valid_target_mask = np.isfinite(target_values)
            if valid_target_mask.sum() == len(X):
                target_rows = all_target_rows[valid_target_mask]
            elif len(all_target_rows) != len(X):
                logger.warning(
                    f"[Rolling Backtest] Could not align target rows exactly: "
                    f"all_targets={len(all_target_rows)}, valid_targets={valid_target_mask.sum()}, samples={len(X)}"
                )
                target_rows = np.arange(len(X), dtype=np.int64) + self.seq_len - 1

        n_samples = len(X)
        val_window = int(n_samples * val_window_ratio)
        if val_window <= 0 or n_samples <= val_window * n_folds:
            logger.warning(
                f"[Rolling Backtest] Not enough samples; skipping backtest: "
                f"n_samples={n_samples}, n_folds={n_folds}, val_window={val_window}"
            )
            return None

        metrics = []
        self._model.eval()
        all_fold_records = []
        for fold in range(n_folds):
            val_start = n_samples - (n_folds - fold) * val_window
            val_end = val_start + val_window
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]
            yids_val = year_ids[val_start:val_end]

            if self._x_scaler is not None:
                x_mean, x_std = self._x_scaler
                X_val = (X_val - x_mean) / x_std
            if self._y_scaler is not None:
                y_mean, y_std = self._y_scaler
                y_val_std = (y_val - y_mean) / y_std
            else:
                y_mean, y_std = 0.0, 1.0
                y_val_std = y_val

            ds = TensorDataset(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val_std.astype(np.float32)),
                torch.from_numpy(yids_val.astype(np.int64)),
            )
            loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)

            preds = []
            ys = []
            with torch.no_grad():
                for bx, by, byid in loader:
                    bx, byid = bx.to(self.device), byid.to(self.device)
                    pred = self._model(bx, self.edge_index, self.edge_weight, year_ids=byid)
                    preds.append(pred.cpu().numpy())
                    ys.append(by.cpu().numpy())

            pred_std = np.concatenate(preds)
            y_true_std = np.concatenate(ys)
            pred_raw = pred_std * y_std + y_mean
            y_true = y_true_std * y_std + y_mean
            pred_raw = np.maximum(pred_raw, 0.0)
            if self._capacity_cap is not None:
                pred_raw = np.minimum(pred_raw, self._capacity_cap)

            denom = np.maximum(np.abs(y_true), 1000.0)
            mape = np.mean(np.abs(pred_raw - y_true) / denom)
            acc = 1.0 - mape
            mae = np.mean(np.abs(pred_raw - y_true))
            rmse = np.sqrt(np.mean((pred_raw - y_true) ** 2))
            ss_res = np.sum((y_true - pred_raw) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0
            bias = np.mean(pred_raw - y_true)

            fold_metrics = {
                "fold": fold + 1,
                "val_start": int(val_start),
                "val_end": int(val_end),
                "accuracy": float(acc),
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse),
                "bias": float(bias),
                "pred_std": float(np.std(pred_raw)),
                "y_std": float(np.std(y_true)),
            }
            metrics.append(fold_metrics)
            logger.info(
                f"[Rolling Backtest][Fold {fold + 1}/{n_folds}] "
                f"samples={val_start}:{val_end}, Accuracy={acc:.4f}, R2={r2:.4f}, "
                f"MAE={mae:.1f}, RMSE={rmse:.1f}, Bias={bias:.1f}, "
                f"pred_std={fold_metrics['pred_std']:.1f}, y_std={fold_metrics['y_std']:.1f}"
            )

            if time_col is not None and time_col in pdf_sorted.columns and len(target_rows) >= val_end:
                fold_target_rows = target_rows[val_start:val_end]
                timestamps = pdf_sorted[time_col].iloc[fold_target_rows].values
                for ts, pr, tr in zip(timestamps, pred_raw, y_true):
                    all_fold_records.append({
                        "fold": fold + 1,
                        "timestamp": ts,
                        "actual": float(tr),
                        "predicted": float(pr),
                        "error": float(pr - tr),
                    })

        if all_fold_records:
            import json
            checkpoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "local", "checkpoints")
            vis_dir = os.path.join(checkpoints_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
            csv_path = os.path.join(vis_dir, "rolling_backtest_predictions.csv")
            pd.DataFrame(all_fold_records).to_csv(csv_path, index=False, encoding="utf-8-sig")
            json_path = os.path.join(vis_dir, "rolling_backtest_predictions.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_fold_records, f, indent=2, default=str)
            logger.info(f"[Rolling Backtest] Saved {len(all_fold_records)} predictions to {csv_path}")

        accs = np.array([m["accuracy"] for m in metrics], dtype=np.float32)
        r2s = np.array([m["r2"] for m in metrics], dtype=np.float32)
        maes = np.array([m["mae"] for m in metrics], dtype=np.float32)
        rmses = np.array([m["rmse"] for m in metrics], dtype=np.float32)
        summary = {
            "folds": metrics,
            "acc_mean": float(accs.mean()),
            "acc_std": float(accs.std()),
            "acc_min": float(accs.min()),
            "acc_max": float(accs.max()),
            "r2_mean": float(r2s.mean()),
            "mae_mean": float(maes.mean()),
            "rmse_mean": float(rmses.mean()),
        }
        logger.info(
            f"[Rolling Backtest] Accuracy={summary['acc_mean']:.4f} ± {summary['acc_std']:.4f} "
            f"(min={summary['acc_min']:.4f}, max={summary['acc_max']:.4f}), "
            f"R2={summary['r2_mean']:.4f}, MAE={summary['mae_mean']:.1f}, RMSE={summary['rmse_mean']:.1f}"
        )
        return summary

    def predict(self, **kwargs):
        """Prediction.

        Expected kwargs:
            feature: pd.DataFrame

        Returns:
            np.ndarray aligned to input length; the first seq_len-1 values are forward-filled.
        """
        feature = None
        for k, v in kwargs.items():
            if "data" in k or "feat" in k or "feature" in k:
                feature = v

        if self._model is None:
            raise RuntimeError("Model is not trained or loaded")
        if feature is None:
            raise ValueError("predict() requires a feature argument")

        feature = feature.reset_index(drop=True)

        # Critical: build samples in sorted order and capture sort indices to remap predictions back
        X, _, sort_idx, year_ids = self._tabular_to_spatiotemporal(
            feature, label_col=None, return_sort_order=True, return_year_ids=True
        )
        if len(X) == 0:
            raise RuntimeError("Prediction data is too short to build one sequence")

        # Standardize using training statistics
        if self._x_scaler is not None:
            x_mean, x_std = self._x_scaler
            if X.shape[-1] != x_mean.shape[-1]:
                raise ValueError(
                    f"Prediction feature dimension {X.shape[-1]} does not match training scaler dimension {x_mean.shape[-1]} 不一致；"
                    f"Delete the old model file and retrain because the feature columns changed."
                )
            X = (X - x_mean) / x_std

        self._model.eval()
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(year_ids.astype(np.int64)))
        loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)

        preds = []
        with torch.no_grad():
            for bx, byid in loader:
                bx, byid = bx.to(self.device), byid.to(self.device)
                pred = self._model(bx, self.edge_index, self.edge_weight, year_ids=byid)
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds)

        # Denormalize to raw scale
        if self._y_scaler is not None:
            y_mean, y_std = self._y_scaler
            preds = preds * y_std + y_mean

        # Physical constraints: power cannot be negative and cannot exceed
        # ~95% of province installed capacity (guards against out-of-range spikes).
        preds = np.maximum(preds, 0.0)
        if self._capacity_cap is not None:
            preds = np.minimum(preds, self._capacity_cap)

        # Distribution diagnostics: compare prediction vs training distributions
        if hasattr(self, '_train_label_dist'):
            pred_mean = float(preds.mean())
            pred_std = float(preds.std())
            pred_min = float(preds.min())
            pred_max = float(preds.max())
            train_label = self._train_label_dist
            logger.info(
                f"[Prediction] Label drift check: pred_mean={pred_mean:.1f} vs train_mean={train_label['mean']:.1f} "
                f"(Δ={pred_mean-train_label['mean']:.1f}), pred_std={pred_std:.1f} vs train_std={train_label['std']:.1f} "
                f"(Δ={pred_std-train_label['std']:.1f}), pred_range=[{pred_min:.1f}, {pred_max:.1f}] "
                f"vs train_range=[{train_label['min']:.1f}, {train_label['max']:.1f}]"
            )
            if pred_std < train_label['std'] * 0.5:
                logger.warning(
                    f"[Prediction] Prediction std ({pred_std:.1f}) is much smaller than training std "
                    f"({train_label['std']:.1f}) — model may be under-predicting variance / compressing amplitude."
                )

        # Feature distribution diagnostics (on standardized X used for prediction)
        if hasattr(self, '_train_feat_dist'):
            feat_mean = float(X.mean())
            feat_std = float(X.std())
            train_feat = self._train_feat_dist
            logger.info(
                f"[Prediction] Feature drift check: pred_feat_mean={feat_mean:.4f} vs train_feat_mean={train_feat['mean']:.4f} "
                f"(Δ={feat_mean-train_feat['mean']:.4f}), pred_feat_std={feat_std:.4f} vs train_feat_std={train_feat['std']:.4f} "
                f"(Δ={feat_std-train_feat['std']:.4f})"
            )

        # Sorted predictions: window i prediction corresponds to sorted row (i + seq_len - 1)
        # Write into sorted array then inverse-permute via sort_idx to original order
        T_sorted = len(sort_idx)
        sorted_preds = np.full(T_sorted, np.nan, dtype=np.float32)
        sorted_preds[self.seq_len - 1 :] = preds

        # Forward-fill the first seq_len-1 positions that lack predictions
        mask = np.isnan(sorted_preds)
        if mask.any() and not mask.all():
            first_valid = np.where(~mask)[0][0]
            sorted_preds[:first_valid] = sorted_preds[first_valid]

        # Inverse permutation: sort_idx[k] is original index → full_preds[sort_idx[k]] = sorted_preds[k]
        full_preds = np.full(len(feature), np.nan, dtype=np.float32)
        full_preds[sort_idx] = sorted_preds

        return full_preds

    def visualize_attention(self, feature, n_samples=4, sample_mode="high_power",
                            save_path=None, figsize=(18, 14)):
        """Extract and visualize GAT attention weights.

        Args:
            feature: pd.DataFrame with raw (unsorted) features
            n_samples: int, number of samples to draw
            sample_mode: str, "high_power" | "low_power" | "random"
            save_path: str or None
            figsize: tuple

        Returns:
            dict: {"attention": alpha, "labels": labels, "samples_info": info}
        """
        if self._model is None:
            raise RuntimeError("Model is not trained or loaded")
        if feature is None:
            raise ValueError("visualize_attention requires feature argument")

        import matplotlib
        matplotlib.use("Agg")  # Use Agg backend for headless environments
        import matplotlib.pyplot as plt

        feature = feature.reset_index(drop=True)

        # 1. Detect label column and build spatiotemporal tensor
        label_col = None
        for col in ['new_energy_wind_ahead', 'new_energy_solar_ahead']:
            if col in feature.columns:
                label_col = col
                break
        X, y, sort_idx, year_ids = self._tabular_to_spatiotemporal(
            feature, label_col=label_col, return_sort_order=True, return_year_ids=True
        )
        if len(X) == 0:
            raise RuntimeError("Data is too short to build sequences")

        # 2. Standardize
        if self._x_scaler is not None:
            x_mean, x_std = self._x_scaler
            X = (X - x_mean) / x_std

        # 3. Labels for sample selection (already raw values from _tabular_to_spatiotemporal)
        if y is not None:
            y_raw = y
        else:
            y_raw = np.zeros(len(X))

        # 4. Select samples
        valid_idx = np.arange(len(X))
        has_label = y is not None and y_raw.std() > 1e-6
        if not has_label:
            logger.warning("[GAT Attention] No valid label column detected in feature; falling back to random sampling")
            sample_mode = "random"
        if sample_mode == "high_power":
            # Select highest-power windows
            chosen = np.argsort(y_raw)[-n_samples:][::-1]
        elif sample_mode == "low_power":
            chosen = np.argsort(y_raw)[:n_samples]
        else:
            rng = np.random.RandomState(42)
            chosen = rng.choice(valid_idx, size=min(n_samples, len(valid_idx)), replace=False)
        chosen = np.array(chosen)

        X_sample = torch.from_numpy(X[chosen]).to(self.device)
        yids_sample = torch.from_numpy(year_ids[chosen].astype(np.int64)).to(self.device)

        # 5. Extract GAT spatial attention
        alpha = self._model.extract_gat_attention(
            X_sample, self.edge_index, self.edge_weight
        )
        # Average over time steps to reduce visual dimensionality
        alpha_time_avg = alpha.mean(axis=1)  # (n_samples, heads, N, N)
        # Average over attention heads
        alpha_avg = alpha_time_avg.mean(axis=1)  # (n_samples, N, N)

        N = len(self._city_list)
        city_list = self._city_list

        # 6. Prepare dense edge-weight matrix for comparison
        edge_weight_dense = np.zeros((N, N), dtype=np.float32)
        if self.edge_weight is not None:
            ei = self.edge_index.cpu().numpy()
            ew = self.edge_weight.cpu().numpy()
            for s, t, w in zip(ei[0], ei[1], ew):
                edge_weight_dense[s, t] = w

        # 7. Plot
        nrows = n_samples
        ncols = 3  # avg attention | per-head attention | edge-weight input
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 gridspec_kw={"width_ratios": [1, 1.2, 1]})
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for row, idx in enumerate(chosen):
            ax1, ax2, ax3 = axes[row]

            # --- Left: averaged attention (heads + time) ---
            attn = alpha_avg[row]  # (N, N)
            im1 = ax1.imshow(attn, cmap="YlOrRd", aspect="auto", vmin=0, vmax=attn.max())
            ax1.set_xticks(range(N))
            ax1.set_yticks(range(N))
            ax1.set_xticklabels(city_list, rotation=60, ha="right", fontsize=7)
            ax1.set_yticklabels(city_list, fontsize=7)
            ax1.set_title(f"Avg Attention (sample #{idx}, power={y_raw[idx]:.0f})", fontsize=9)
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            # Highlight Yancheng (index 8) row/col to check if it dominates messaging
            yc_idx = city_list.index("Yancheng") if "Yancheng" in city_list else -1
            if yc_idx >= 0:
                # Highlight Yancheng column (source) and row (target)
                for ax in [ax1]:
                    ax.axvline(x=yc_idx - 0.5, color="blue", linewidth=1.5, linestyle="--")
                    ax.axhline(y=yc_idx - 0.5, color="blue", linewidth=1.5, linestyle="--")

            # --- Center: per-head attention grid ---
            n_heads = alpha_time_avg.shape[1]
            # Time-averaged attention per head
            head_grid = alpha_time_avg[row]  # (heads, N, N)
            # Concatenate all heads into one large image
            head_combined = np.concatenate(list(head_grid), axis=1)  # (N, heads*N)
            im2 = ax2.imshow(head_combined, cmap="YlOrRd", aspect="auto", vmin=0)
            ax2.set_xticks([N * h + N // 2 for h in range(n_heads)])
            ax2.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=8)
            ax2.set_yticks(range(N))
            ax2.set_yticklabels(city_list, fontsize=7)
            ax2.set_title("Per-Head Attention (time-avg)", fontsize=9)
            plt.colorbar(im2, ax=ax2, fraction=0.046)
            if yc_idx >= 0:
                ax2.axhline(y=yc_idx - 0.5, color="blue", linewidth=1.5, linestyle="--")

            # --- Right: input edge_weight matrix ---
            im3 = ax3.imshow(edge_weight_dense, cmap="Blues", aspect="auto")
            ax3.set_xticks(range(N))
            ax3.set_yticks(range(N))
            ax3.set_xticklabels(city_list, rotation=60, ha="right", fontsize=7)
            ax3.set_yticklabels(city_list, fontsize=7)
            ax3.set_title("Input Edge Weight", fontsize=9)
            plt.colorbar(im3, ax=ax3, fraction=0.046)
            if yc_idx >= 0:
                ax3.axvline(x=yc_idx - 0.5, color="red", linewidth=1.5, linestyle="--")
                ax3.axhline(y=yc_idx - 0.5, color="red", linewidth=1.5, linestyle="--")

        fig.suptitle(
            f"GAT Attention Visualization (mode={sample_mode}, n={n_samples})\n"
            f"Blue dashed = Yancheng row/col; Red dashed = Yancheng in edge_weight",
            fontsize=11, y=1.02
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"[GAT Attention] Visualization saved: {save_path}")
        plt.close(fig)

        # 8. Temporal attention visualization (Informer last-layer self-attention)
        try:
            temporal_attn = self._model.extract_temporal_attention(
                X_sample, self.edge_index, self.edge_weight, year_ids=yids_sample
            )  # (n_samples, T, T)
            self._visualize_temporal_attention(temporal_attn, chosen, y_raw, sample_mode, save_path)
        except Exception as e:
            logger.warning(f"[Informer Temporal Attention] Visualization failed (non-critical): {e}")

        # 9. Automatic diagnostic: Yancheng monopoly
        info = []
        for row, idx in enumerate(chosen):
            attn = alpha_avg[row]  # (N, N), row=target, col=source
            # Yancheng column share per target node
            yc_idx = city_list.index("Yancheng") if "Yancheng" in city_list else -1
            if yc_idx < 0:
                continue
            yc_share_per_target = attn[:, yc_idx]  # (N,)
            avg_share = float(yc_share_per_target.mean())
            max_share = float(yc_share_per_target.max())
            # Yancheng self-loop share
            self_loop_share = float(attn[yc_idx, yc_idx])
            info.append({
                "sample_idx": int(idx),
                "power": float(y_raw[idx]),
                "yancheng_avg_attention_share": round(avg_share, 4),
                "yancheng_max_attention_share": round(max_share, 4),
                "yancheng_self_loop_share": round(self_loop_share, 4),
            })
            logger.info(
                f"[GAT Attention][Sample {idx}] power={y_raw[idx]:.0f}, "
                f"Yancheng avg attention share={avg_share:.4f}, max={max_share:.4f}, "
                f"self-loop share={self_loop_share:.4f}"
            )

        # Summary
        if info:
            avg_across_samples = np.mean([d["yancheng_avg_attention_share"] for d in info])
            logger.info(
                f"[GAT Attention] 诊断: Yancheng avg attention share={avg_across_samples:.4f} "
                f"(>0.5 means strong dominance, 0.3-0.5 moderate dominance, <0.3 diffuse)"
            )

        return {
            "attention": alpha,
            "labels": y_raw[chosen] if y_raw is not None else None,
            "samples_info": info,
        }

    def _visualize_temporal_attention(self, temporal_attn, chosen, y_raw, sample_mode, save_path=None):
        """Visualize Informer last-layer temporal attention and output recency diagnostics."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if temporal_attn is None or len(temporal_attn) == 0:
            return

        n_samples, T, _ = temporal_attn.shape
        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 3.6 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        temporal_info = []
        lag_axis = np.arange(T) - (T - 1)  # -95 ... 0, 0 is the prediction timestep
        for row, idx in enumerate(chosen):
            attn = temporal_attn[row]
            # Last query attention on historical keys is the most direct temporal dependency
            last_query_attn = attn[-1]
            recent_24 = float(last_query_attn[-24:].sum()) if T >= 24 else float(last_query_attn.sum())
            recent_12 = float(last_query_attn[-12:].sum()) if T >= 12 else float(last_query_attn.sum())
            entropy = float(-(last_query_attn * np.log(last_query_attn + 1e-12)).sum())
            top_lags = lag_axis[np.argsort(last_query_attn)[-5:][::-1]].tolist()
            temporal_info.append({
                "sample_idx": int(idx),
                "power": float(y_raw[idx]),
                "recent_12_share": round(recent_12, 4),
                "recent_24_share": round(recent_24, 4),
                "entropy": round(entropy, 4),
                "top_lags": [int(x) for x in top_lags],
            })

            ax1, ax2 = axes[row]
            im = ax1.imshow(attn, cmap="viridis", aspect="auto", vmin=0)
            ax1.set_title(f"Temporal Attention Matrix (sample #{idx}, power={y_raw[idx]:.0f})", fontsize=9)
            ax1.set_xlabel("Key time index")
            ax1.set_ylabel("Query time index")
            plt.colorbar(im, ax=ax1, fraction=0.046)

            ax2.plot(lag_axis, last_query_attn, linewidth=1.5)
            ax2.axvspan(-23, 0, color="orange", alpha=0.15, label="recent 24 steps")
            ax2.set_title(
                f"Last-step attention: recent12={recent_12:.2f}, recent24={recent_24:.2f}, entropy={entropy:.2f}",
                fontsize=9,
            )
            ax2.set_xlabel("Lag steps (0=current, -95=oldest)")
            ax2.set_ylabel("Attention weight")
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)

            logger.info(
                f"[Informer Temporal Attention][Sample {idx}] power={y_raw[idx]:.0f}, "
                f"recent12={recent_12:.4f}, recent24={recent_24:.4f}, "
                f"entropy={entropy:.4f}, top_lags={top_lags}"
            )

        avg_recent_24 = np.mean([d["recent_24_share"] for d in temporal_info])
        logger.info(
            f"[Informer Temporal Attention] diagnostic: avg recent-24 attention share={avg_recent_24:.4f} "
            f"(>0.7 means over-reliance on recent history, <0.4 means longer history is used)"
        )

        fig.suptitle(f"Informer Temporal Attention (mode={sample_mode}, n={n_samples})", fontsize=11, y=1.02)
        fig.tight_layout()
        if save_path:
            root, ext = os.path.splitext(save_path)
            temporal_path = f"{root}_temporal{ext or '.png'}"
            fig.savefig(temporal_path, dpi=150, bbox_inches="tight")
            logger.info(f"[Informer Temporal Attention] Visualization saved: {temporal_path}")
        plt.close(fig)
        return temporal_info

    def save(self, **kwargs):
        """Save model weights, graph structure, and hyperparameters."""
        model_name = "model.pt"
        for k, v in kwargs.items():
            if "model_name" in k:
                model_name = v

        if self._model is None:
            raise RuntimeError("Model is not trained; cannot save")

        state = {
            "model_state_dict": self._model.state_dict(),
            "edge_index": self.edge_index.cpu(),
            "edge_weight": self.edge_weight.cpu(),
            "hyperparams": {
                "label_type": self.label_type,
                "seq_len": self.seq_len,
                "d_gat": self.d_gat,
                "gat_heads": self.gat_heads,
                "d_model": self.d_model,
                "informer_layers": self.informer_layers,
                "informer_heads": self.informer_heads,
                "d_ff": self.d_ff,
                "dropout": self.dropout,
                "node_feat_dim": self._node_feat_dim,
                "drop_edge": self.drop_edge,
                "temporal_noise": self.temporal_noise,
                "weight_decay": self.weight_decay,
                "clip_grad": self.clip_grad,
            },
            "city_list": self._city_list,
            "capacity_dict": self._capacity_dict,
            "weather_cols": self._weather_cols,
            "x_scaler": (
                self._x_scaler[0].tolist() if self._x_scaler is not None else None,
                self._x_scaler[1].tolist() if self._x_scaler is not None else None,
            ),
            "y_scaler": self._y_scaler,
            "global_feat_cols": self._global_feat_cols,
            "loss_history": self.loss_history,
        }
        torch.save(state, model_name)
        logger.info(f"[GAT+Informer] Model saved: {model_name}")

    def load(self, **kwargs):
        """Load model weights, graph structure, and hyperparameters."""
        model_name = "model.pt"
        for k, v in kwargs.items():
            if "model_name" in k:
                model_name = v

        if not os.path.exists(model_name):
            raise FileNotFoundError(f"Model file does not exist: {model_name}")

        state = torch.load(model_name, map_location=self.device)

        # Restore hyperparameters
        hparams = state["hyperparams"]
        self.label_type = hparams["label_type"]
        self.seq_len = hparams["seq_len"]
        self.d_gat = hparams["d_gat"]
        self.gat_heads = hparams["gat_heads"]
        self.d_model = hparams["d_model"]
        self.informer_layers = hparams["informer_layers"]
        self.informer_heads = hparams["informer_heads"]
        self.d_ff = hparams["d_ff"]
        self.dropout = hparams["dropout"]
        self.drop_edge = hparams.get("drop_edge", self.drop_edge)
        self.temporal_noise = hparams.get("temporal_noise", self.temporal_noise)
        self.weight_decay = hparams.get("weight_decay", self.weight_decay)
        self.clip_grad = hparams.get("clip_grad", self.clip_grad)
        self._node_feat_dim = hparams["node_feat_dim"]
        self._city_list = state["city_list"]
        self._capacity_dict = state["capacity_dict"]
        self._weather_cols = state["weather_cols"]

        # Restore scalers
        x_scaler_state = state.get("x_scaler", (None, None))
        if x_scaler_state[0] is not None and x_scaler_state[1] is not None:
            x_mean = np.array(x_scaler_state[0], dtype=np.float32)
            x_std = np.array(x_scaler_state[1], dtype=np.float32)
            self._x_scaler = (x_mean, x_std)
        else:
            self._x_scaler = None
        self._y_scaler = state.get("y_scaler", (0.0, 1.0))
        self._global_feat_cols = state.get("global_feat_cols", None)
        self.loss_history = state.get("loss_history", None)

        # Restore graph structure
        self.edge_index = state["edge_index"].to(self.device)
        self.edge_weight = state["edge_weight"].to(self.device)

        # Initialize and load network
        self._model = SpatiotemporalGATInformer(
            num_nodes=len(self._city_list),
            node_feat_dim=self._node_feat_dim,
            d_gat=self.d_gat,
            gat_heads=self.gat_heads,
            d_model=self.d_model,
            informer_layers=self.informer_layers,
            informer_heads=self.informer_heads,
            d_ff=self.d_ff,
            seq_len=self.seq_len,
            dropout=self.dropout,
            drop_edge=self.drop_edge,
            temporal_noise=self.temporal_noise,
        ).to(self.device)
        self._model.load_state_dict(state["model_state_dict"])

        logger.info(f"[GAT+Informer] Model loaded: {model_name}")
