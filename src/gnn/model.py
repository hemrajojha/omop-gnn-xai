import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import to_dense_batch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph schema (must match build_patient_graphs.py and pyhealth_sample_to_graph)
# ---------------------------------------------------------------------------

_NODE_TYPES = ["visit", "condition", "procedure", "drug"]

_EDGE_TYPES = [
    ("condition", "r_has_condition", "visit"),
    ("procedure", "r_has_procedure", "visit"),
    ("drug",      "r_has_drug",      "visit"),
    ("visit",     "temporal_next",   "visit"),
]

_METADATA = (_NODE_TYPES, _EDGE_TYPES)


# ---------------------------------------------------------------------------
# KG embedding helpers (unchanged)
# ---------------------------------------------------------------------------

def _build_domain_embedding_table(
    domain: str,
    concept_vocab: dict,
    kg_vocab: dict,
    kg_matrix: np.ndarray,
    kg_embed_dim: int,
) -> tuple[np.ndarray, int, int]:
    domain_vocab = concept_vocab.get(domain, {})
    if not domain_vocab:
        return np.zeros((0, kg_embed_dim), dtype=np.float32), 0, 0

    vocab_size = max(int(v) for v in domain_vocab.values()) + 1
    rng = np.random.default_rng(42)
    table = rng.normal(0, 0.01, (vocab_size, kg_embed_dim)).astype(np.float32)

    n_matched = 0
    for cid_str, vidx in domain_vocab.items():
        kg_idx = kg_vocab.get(cid_str)
        if kg_idx is not None:
            table[int(vidx)] = kg_matrix[int(kg_idx)]
            n_matched += 1

    return table, n_matched, vocab_size


def load_kg_embeddings(
    concept_vocab_path: str | Path,
    kg_vocab_path: str | Path,
    kg_matrix_path: str | Path,
) -> dict[str, tuple[np.ndarray, int]]:
    with open(concept_vocab_path) as f:
        concept_vocab = json.load(f)
    with open(kg_vocab_path) as f:
        kg_vocab = json.load(f)

    kg_matrix = np.load(kg_matrix_path).astype(np.float32)
    kg_embed_dim = kg_matrix.shape[1]
    log.info("KG matrix shape: %s", kg_matrix.shape)

    results = {}
    for domain in ("condition", "procedure", "drug"):
        table, n_matched, vocab_size = _build_domain_embedding_table(
            domain, concept_vocab, kg_vocab, kg_matrix, kg_embed_dim
        )
        log.info(
            "  %s: vocab_size=%d, kg_matched=%d (%.1f%%)",
            domain, vocab_size, n_matched,
            100 * n_matched / max(vocab_size, 1),
        )
        results[domain] = (table, n_matched)

    return results


# ---------------------------------------------------------------------------
# Positional / structural encoding helpers
# ---------------------------------------------------------------------------

def _sinusoidal_lpe(positions: torch.Tensor, dim: int) -> torch.Tensor:
    N = positions.size(0)
    pe = torch.zeros(N, dim, device=positions.device)
    pos = positions.float().unsqueeze(1)                              # [N, 1]
    half = dim // 2
    div = torch.exp(
        torch.arange(half, device=positions.device).float()
        * -(math.log(10000.0) / half)
    )                                                                  # [half]
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe                                                          # [N, dim]


def _metapath_rw_se(
    data: HeteroData,
    num_visits: int,
    se_dim: int,
    device: torch.device,
) -> torch.Tensor:
    fwd_keys = [
        ("visit", "has_condition", "condition"),
        ("visit", "has_procedure", "procedure"),
        ("visit", "has_drug",      "drug"),
    ]

    # Degree of each visit in each domain metapath (normalised to [0,1])
    degs = []
    for fwd_key in fwd_keys:
        if fwd_key in data.edge_types and data[fwd_key].edge_index.size(1) > 0:
            ei = data[fwd_key].edge_index                   # [2, E]
            deg = torch.zeros(num_visits, device=device)
            deg.scatter_add_(
                0,
                ei[0].to(device),
                torch.ones(ei.size(1), device=device),
            )
            deg = deg / deg.max().clamp(min=1.0)
        else:
            deg = torch.zeros(num_visits, device=device)
        degs.append(deg)

    # Build se_dim features: cycle [cond^1, proc^1, drug^1, cond^2, proc^2, …]
    features = []
    for i in range(se_dim):
        domain_idx = i % 3
        power      = i // 3 + 1
        v = degs[domain_idx].pow(power)
        v = v / v.max().clamp(min=1e-8)
        features.append(v)

    return torch.stack(features, dim=1)      # [num_visits, se_dim]


# ---------------------------------------------------------------------------
# PatientGNN
# ---------------------------------------------------------------------------

class PatientGNN(nn.Module):
    def __init__(
        self,
        visit_feat_dim: int,
        hidden_dim: int,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.3,
        pe_dim: int = 4,
        kg_embed_dim: int = 128,
        # KG embedding tables (None in no-KG mode)
        cond_embed_table: Optional[np.ndarray] = None,
        proc_embed_table: Optional[np.ndarray] = None,
        drug_embed_table: Optional[np.ndarray] = None,
        freeze_kg: bool = True,
        # No-KG mode
        no_kg: bool = False,
        cond_vocab_size: int = 0,
        proc_vocab_size: int = 0,
        drug_vocab_size: int = 0,
        # Diagnosis task
        num_labels: int = 0,
        # Dual-branch fusion weight
        alpha: float = 0.8,
    ):
        super().__init__()

        assert hidden_dim % nhead == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})"
        )

        self.hidden_dim = hidden_dim
        self.pe_dim     = pe_dim
        self.dropout    = dropout
        self.num_labels = num_labels
        self.alpha      = alpha

        # ---- Concept embeddings ------------------------------------------------
        self.cond_embed = self._make_embedding(
            cond_embed_table, cond_vocab_size, kg_embed_dim, freeze_kg, no_kg
        )
        self.proc_embed = self._make_embedding(
            proc_embed_table, proc_vocab_size, kg_embed_dim, freeze_kg, no_kg
        )
        self.drug_embed = self._make_embedding(
            drug_embed_table, drug_vocab_size, kg_embed_dim, freeze_kg, no_kg
        )

        # ---- Input projections -------------------------------------------------
        # Visit: raw features + Laplacian PE (pe_dim) + MetaPath SE (pe_dim)
        self.visit_proj   = nn.Sequential(
            nn.Linear(visit_feat_dim + 2 * pe_dim, hidden_dim), nn.ReLU()
        )
        self.concept_proj = nn.Linear(kg_embed_dim, hidden_dim)

        # ---- HGTConv layers (replaces SAGEConv) --------------------------------
        # HGTConv: relation-specific attention + type-specific projections
        # (Hu et al., "Heterogeneous Graph Transformer", WWW 2020)
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, _METADATA, heads=nhead)
            for _ in range(num_gnn_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # ---- Graph branch: Transformer over GNN-enriched visit sequence --------
        # (replaces GRU; global self-attention over visits)
        graph_tf_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_transformer = nn.TransformerEncoder(
            graph_tf_layer, num_layers=num_transformer_layers
        )

        # ---- Sequence branch: Transformer over raw visit features --------------
        # Parallel branch; receives no GNN output (pure sequence signal)
        self.seq_proj = nn.Sequential(
            nn.Linear(visit_feat_dim, hidden_dim), nn.ReLU()
        )
        seq_tf_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_transformer = nn.TransformerEncoder(
            seq_tf_layer, num_layers=num_transformer_layers
        )

        # ---- Prediction heads --------------------------------------------------
        if num_labels > 0:
            self.graph_head = nn.Linear(hidden_dim, num_labels)
            self.seq_head   = nn.Linear(hidden_dim, num_labels)
        else:
            self.graph_mort_head = nn.Linear(hidden_dim, 1)
            self.graph_read_head = nn.Linear(hidden_dim, 1)
            self.seq_mort_head   = nn.Linear(hidden_dim, 1)
            self.seq_read_head   = nn.Linear(hidden_dim, 1)

        self._init_weights()

    # -----------------------------------------------------------------------

    @staticmethod
    def _make_embedding(
        table: Optional[np.ndarray],
        vocab_size: int,
        embed_dim: int,
        freeze: bool,
        no_kg: bool,
    ) -> nn.Embedding:
        if not no_kg and table is not None and table.shape[0] > 0:
            weight = torch.tensor(table, dtype=torch.float)
            return nn.Embedding.from_pretrained(weight, freeze=freeze)
        return nn.Embedding(max(vocab_size, 1), embed_dim)

    def _init_weights(self):
        if self.num_labels > 0:
            heads = [self.graph_head, self.seq_head, self.concept_proj]
        else:
            heads = [
                self.graph_mort_head, self.graph_read_head,
                self.seq_mort_head,   self.seq_read_head,
                self.concept_proj,
            ]
        for m in heads:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # PE helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _visit_positions(batch_vec: torch.Tensor) -> torch.Tensor:
        _, counts = torch.unique_consecutive(batch_vec, return_counts=True)
        return torch.cat([
            torch.arange(c, device=batch_vec.device) for c in counts
        ])

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self, data: HeteroData, task: str = "mort_read"
    ) -> dict[str, torch.Tensor]:
        device      = data["visit"].x.device
        raw_visit   = data["visit"].x                 # [N_v, visit_feat_dim]
        num_visits  = raw_visit.size(0)

        batch_vec = (
            data["visit"].batch
            if hasattr(data["visit"], "batch")
            else raw_visit.new_zeros(num_visits, dtype=torch.long)
        )

        # ---- 1. Positional / structural encodings --------------------------
        positions = self._visit_positions(batch_vec)           # [N_v]
        lpe = _sinusoidal_lpe(positions, self.pe_dim)          # [N_v, pe_dim]
        rws = _metapath_rw_se(                                 # [N_v, pe_dim]
            data, num_visits, self.pe_dim, device
        )

        # ---- 2. Build x_dict -----------------------------------------------
        visit_in = torch.cat([raw_visit, lpe, rws], dim=-1)   # [N_v, feat+2*pe]
        x_dict   = {"visit": self.visit_proj(visit_in)}

        for node_type, embed in [
            ("condition", self.cond_embed),
            ("procedure", self.proc_embed),
            ("drug",      self.drug_embed),
        ]:
            if node_type in data.node_types and data[node_type].x.size(0) > 0:
                idx = data[node_type].x[:, 0].long().clamp(
                    0, embed.weight.size(0) - 1
                )
                x_dict[node_type] = F.relu(self.concept_proj(embed(idx)))
            else:
                x_dict[node_type] = x_dict["visit"].new_zeros(
                    (0, self.hidden_dim)
                )

        # ---- 3. Build edge_index_dict for HGTConv --------------------------
        edge_index_dict = self._build_edge_index_dict(data)

        # ---- 4. HGTConv layers ---------------------------------------------
        for conv, norm in zip(self.convs, self.norms):
            out    = conv(x_dict, edge_index_dict)
            v_out  = out.get("visit", x_dict["visit"])
            v_out  = F.dropout(F.relu(v_out), p=self.dropout, training=self.training)
            x_dict["visit"] = norm(v_out + x_dict["visit"])   # residual

        visit_emb = x_dict["visit"]                            # [N_v, H]

        # ---- 5. Pack into dense [B, max_T, H] ------------------------------
        visit_padded, mask = to_dense_batch(visit_emb, batch_vec)
        key_padding_mask   = ~mask                             # True = padded pos

        # ---- 6. Graph branch: Transformer over GNN-enriched visits ---------
        graph_out = self.graph_transformer(
            visit_padded, src_key_padding_mask=key_padding_mask
        )   # [B, T, H]

        # ---- 7. Sequence branch: Transformer over raw visit features -------
        raw_padded, _ = to_dense_batch(
            self.seq_proj(raw_visit), batch_vec
        )   # [B, T, H]
        seq_out = self.seq_transformer(
            raw_padded, src_key_padding_mask=key_padding_mask
        )   # [B, T, H]

        # ---- 8. Dual-branch fusion + prediction heads ----------------------
        B      = graph_out.size(0)
        device = graph_out.device

        if task == "diagnosis":
            # Last non-padded visit per patient → [B, H]
            seq_lens = mask.sum(dim=1) - 1
            idx      = torch.arange(B, device=device)
            g_last   = graph_out[idx, seq_lens]     # [B, H]
            s_last   = seq_out[idx, seq_lens]       # [B, H]

            graph_logits = self.graph_head(g_last)  # [B, num_labels]
            seq_logits   = self.seq_head(s_last)    # [B, num_labels]
            logits = self.alpha * graph_logits + (1 - self.alpha) * seq_logits
            return {"diagnosis": logits}

        else:  # mort_read — per-visit predictions
            g_visits = graph_out[mask]              # [N_v, H]
            s_visits = seq_out[mask]                # [N_v, H]

            graph_mort = self.graph_mort_head(g_visits).squeeze(-1)
            graph_read = self.graph_read_head(g_visits).squeeze(-1)
            seq_mort   = self.seq_mort_head(s_visits).squeeze(-1)
            seq_read   = self.seq_read_head(s_visits).squeeze(-1)

            return {
                "mortality":   self.alpha * graph_mort + (1 - self.alpha) * seq_mort,
                "readmission": self.alpha * graph_read + (1 - self.alpha) * seq_read,
            }

    # -----------------------------------------------------------------------

    @staticmethod
    def _build_edge_index_dict(data: HeteroData) -> dict:
        edge_index_dict = {}

        reverse_map = {
            ("visit", "has_condition", "condition"): ("condition", "r_has_condition", "visit"),
            ("visit", "has_procedure", "procedure"): ("procedure", "r_has_procedure", "visit"),
            ("visit", "has_drug",      "drug"):      ("drug",      "r_has_drug",      "visit"),
        }
        for fwd_key, rev_key in reverse_map.items():
            if fwd_key in data.edge_types:
                ei = data[fwd_key].edge_index
                if ei.size(1) > 0:
                    edge_index_dict[rev_key] = ei.flip(0)

        temporal_key = ("visit", "temporal_next", "visit")
        if temporal_key in data.edge_types:
            ei = data[temporal_key].edge_index
            if ei.size(1) > 0:
                edge_index_dict[temporal_key] = ei

        return edge_index_dict

    # -----------------------------------------------------------------------
    # Explainability
    # -----------------------------------------------------------------------

    def explain(
        self,
        data: HeteroData,
        label_idx: int,
        task: str = "diagnosis",
        top_k: int = 10,
    ) -> dict:
        self.eval()

        # Enable gradients on concept node features
        embed_params = {}
        for domain, embed in [("condition", self.cond_embed),
                               ("procedure", self.proc_embed),
                               ("drug",      self.drug_embed)]:
            if domain in data.node_types and data[domain].x.size(0) > 0:
                idx = data[domain].x[:, 0].long().clamp(0, embed.weight.size(0) - 1)
                emb = embed(idx)                     # [N_domain, kg_embed_dim]
                emb.retain_grad()
                embed_params[domain] = (emb, idx)

        # Monkey-patch forward to use pre-computed embeddings with grad
        device = data["visit"].x.device
        num_visits = data["visit"].x.size(0)
        batch_vec = data["visit"].x.new_zeros(num_visits, dtype=torch.long)
        positions = self._visit_positions(batch_vec)
        lpe = _sinusoidal_lpe(positions, self.pe_dim)
        rws = _metapath_rw_se(data, num_visits, self.pe_dim, device)

        visit_in = torch.cat([data["visit"].x, lpe, rws], dim=-1)
        x_dict = {"visit": self.visit_proj(visit_in)}

        for domain, embed in [("condition", self.cond_embed),
                               ("procedure", self.proc_embed),
                               ("drug",      self.drug_embed)]:
            if domain in embed_params:
                emb, _ = embed_params[domain]
                x_dict[domain] = F.relu(self.concept_proj(emb))
            else:
                x_dict[domain] = x_dict["visit"].new_zeros((0, self.hidden_dim))

        edge_index_dict = self._build_edge_index_dict(data)
        for conv, norm in zip(self.convs, self.norms):
            out   = conv(x_dict, edge_index_dict)
            v_out = out.get("visit", x_dict["visit"])
            v_out = F.dropout(F.relu(v_out), p=self.dropout, training=self.training)
            x_dict["visit"] = norm(v_out + x_dict["visit"])

        visit_emb = x_dict["visit"]
        visit_padded, mask = to_dense_batch(visit_emb, batch_vec)
        key_padding_mask = ~mask

        graph_out = self.graph_transformer(
            visit_padded, src_key_padding_mask=key_padding_mask
        )
        raw_padded, _ = to_dense_batch(self.seq_proj(data["visit"].x), batch_vec)
        seq_out = self.seq_transformer(
            raw_padded, src_key_padding_mask=key_padding_mask
        )

        seq_lens = mask.sum(dim=1) - 1
        g_last = graph_out[0, seq_lens[0]]
        s_last = seq_out[0, seq_lens[0]]
        graph_logits = self.graph_head(g_last)
        seq_logits   = self.seq_head(s_last)
        logits = self.alpha * graph_logits + (1 - self.alpha) * seq_logits

        # Backprop through label_idx
        logits[label_idx].backward()

        # Collect attribution scores
        results = {
            "logit":       logits[label_idx].item(),
            "probability": torch.sigmoid(logits[label_idx]).item(),
        }
        for domain in ("condition", "procedure", "drug"):
            if domain in embed_params:
                emb, node_indices = embed_params[domain]
                if emb.grad is not None:
                    scores = emb.grad.abs().sum(dim=-1)   # [N_domain]
                    k = min(top_k, scores.size(0))
                    top_vals, top_pos = scores.topk(k)
                    results[domain] = [
                        (node_indices[i].item(), top_vals[j].item())
                        for j, i in enumerate(top_pos)
                    ]
                else:
                    results[domain] = []
            else:
                results[domain] = []

        return results

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        # KG paths (required unless no_kg=True)
        concept_vocab_path: Optional[str | Path] = None,
        kg_vocab_path:      Optional[str | Path] = None,
        kg_matrix_path:     Optional[str | Path] = None,
        # Architecture
        hidden_dim:             int   = 128,
        num_gnn_layers:         int   = 2,
        num_transformer_layers: int   = 2,
        nhead:                  int   = 4,
        dropout:                float = 0.3,
        pe_dim:                 int   = 4,
        freeze_kg:              bool  = True,
        visit_feat_dim:         int   = 7,
        alpha:                  float = 0.8,
        # No-KG mode
        no_kg:          bool = False,
        cond_vocab_size: int = 0,
        proc_vocab_size: int = 0,
        drug_vocab_size: int = 0,
        # Task
        num_labels:  int = 0,
        kg_embed_dim: int = 128,
    ) -> "PatientGNN":
        if no_kg:
            return cls(
                visit_feat_dim=visit_feat_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                num_transformer_layers=num_transformer_layers,
                nhead=nhead,
                dropout=dropout,
                pe_dim=pe_dim,
                kg_embed_dim=kg_embed_dim,
                no_kg=True,
                cond_vocab_size=cond_vocab_size,
                proc_vocab_size=proc_vocab_size,
                drug_vocab_size=drug_vocab_size,
                num_labels=num_labels,
                alpha=alpha,
            )

        # KG mode
        kg_tables    = load_kg_embeddings(concept_vocab_path, kg_vocab_path, kg_matrix_path)
        cond_table, _ = kg_tables["condition"]
        proc_table, _ = kg_tables["procedure"]
        drug_table, _ = kg_tables["drug"]
        actual_kg_dim = cond_table.shape[1] if cond_table.shape[0] > 0 else kg_embed_dim

        return cls(
            visit_feat_dim=visit_feat_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            nhead=nhead,
            dropout=dropout,
            pe_dim=pe_dim,
            kg_embed_dim=actual_kg_dim,
            cond_embed_table=cond_table,
            proc_embed_table=proc_table,
            drug_embed_table=drug_table,
            freeze_kg=freeze_kg,
            no_kg=False,
            num_labels=num_labels,
            alpha=alpha,
        )
