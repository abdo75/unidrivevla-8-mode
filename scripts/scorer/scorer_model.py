"""Learned PDMS scorer: encode a candidate trajectory, attend over the scene tokens
(phi_s), and predict the PDMS sub-scores in [0, 1].

Keeping the scene tokens (rather than mean-pooling) lets the candidate query attend
to where objects / drivable area are -- the information the collision / drivable-area
sub-scores depend on.
"""
import torch
import torch.nn as nn


class Scorer(nn.Module):
    def __init__(self, c_in, k, t_horizon=6, d=128, n_heads=4, pool="attn"):
        """c_in: phi_s channel dim; k: number of sub-scores; t_horizon: waypoints.
        pool selects how the candidate reads the scene (ablation in sec:ablation):
          'attn'     - candidate query cross-attends over the scene tokens (default),
          'mean'     - mean-pool the scene tokens, concatenate with the candidate,
          'trajonly' - ignore the scene entirely (candidate geometry only)."""
        super().__init__()
        self.pool = pool
        self.traj_enc = nn.Sequential(
            nn.Flatten(), nn.Linear(t_horizon * 2, d), nn.ReLU(), nn.Linear(d, d)
        )
        if pool == "attn":
            self.kv = nn.Linear(c_in, d)
            self.attn = nn.MultiheadAttention(d, num_heads=n_heads, batch_first=True)
        elif pool == "mean":
            self.scene_proj = nn.Linear(c_in, d)
            self.fuse = nn.Linear(2 * d, d)
        elif pool != "trajonly":
            raise ValueError(f"unknown pool {pool!r}")
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, k))

    def forward(self, phi, traj, key_padding_mask=None):
        """phi: (B, T_s, c_in) scene tokens; traj: (B, T, 2) candidate waypoints.
        key_padding_mask: (B, T_s) bool, True where padded (ignored by attention)."""
        q = self.traj_enc(traj)                          # (B, d)
        if self.pool == "attn":
            kv = self.kv(phi)                            # (B, T_s, d)
            ctx, _ = self.attn(q.unsqueeze(1), kv, kv, key_padding_mask=key_padding_mask)
            z = ctx.squeeze(1)                           # (B, d)
        elif self.pool == "mean":
            if key_padding_mask is not None:             # masked mean over valid tokens
                w = (~key_padding_mask).float().unsqueeze(-1)
                scene = (phi * w).sum(1) / w.sum(1).clamp(min=1.0)
            else:
                scene = phi.mean(1)
            z = self.fuse(torch.cat([q, self.scene_proj(scene)], dim=-1))
        else:                                            # trajonly
            z = q
        return torch.sigmoid(self.head(z))               # (B, k) in [0,1]