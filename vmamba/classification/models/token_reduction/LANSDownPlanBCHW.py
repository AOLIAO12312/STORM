import torch
import torch.nn as nn
import torch.nn.functional as F

def _normalize_feat(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))

def _similarity(a, b, distance='cosine'):
    if distance == 'cosine':
        return (_normalize_feat(a) * _normalize_feat(b)).sum(dim=-1)
    elif distance == 'l2':
        return -((a - b) ** 2).sum(dim=-1)
    else:
        raise NotImplementedError

def _make_frac_grid(H_in, W_in, H_out, W_out, align_corners=False, device=None):
    if align_corners:
        yy = torch.linspace(0, H_in - 1, H_out, device=device)
        xx = torch.linspace(0, W_in - 1, W_out, device=device)
    else:
        yy = (torch.arange(H_out, device=device) + 0.5) * (H_in / H_out) - 0.5
        xx = (torch.arange(W_out, device=device) + 0.5) * (W_in / W_out) - 0.5
    Yc = yy[:, None].expand(H_out, W_out)
    Xc = xx[None, :].expand(H_out, W_out)
    return Yc, Xc

def _neighbors_2x2(Yc, Xc):
    y0 = torch.floor(Yc).to(torch.long)
    x0 = torch.floor(Xc).to(torch.long)
    oy = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=Yc.device)
    ox = torch.tensor([0, 1, 0, 1], dtype=torch.long, device=Yc.device)
    return y0, x0, oy, ox

def _neighbors_k(Yc, Xc, k):
    assert k % 2 == 1, "k 应为奇数"
    half = k // 2
    yc0 = torch.round(Yc).to(torch.long)
    xc0 = torch.round(Xc).to(torch.long)
    offs = torch.arange(-half, half + 1, device=Yc.device)
    oy, ox = torch.meshgrid(offs, offs, indexing='ij')
    oy = oy.reshape(-1).to(torch.long)
    ox = ox.reshape(-1).to(torch.long)
    return yc0, xc0, oy, ox

def _bilinear_base_weights(Yc, Xc, y0, x0):
    dy = (Yc - y0.float()).clamp(0, 1)
    dx = (Xc - x0.float()).clamp(0, 1)
    w00 = (1 - dy) * (1 - dx)
    w01 = (1 - dy) * dx
    w10 = dy * (1 - dx)
    w11 = dy * dx
    return torch.stack([w00, w01, w10, w11], dim=-1)  # [..., 4]

def _gaussian_base_weights(Yc, Xc, ys, xs, sigma=0.5):
    dy2 = (Yc[..., None] - ys.float()) ** 2
    dx2 = (Xc[..., None] - xs.float()) ** 2
    return torch.exp(-(dy2 + dx2) / (2 * (sigma ** 2)))  # [..., K]

class LANSDownPlanBCHW(nn.Module):
    """
    Leverage-Aware Nyström Selection (LANS-Down) 规划器（BCHW）。

    思路：
      - 对每个 2x2 局部候选 {x1,x2,x3,x4}，在“注意力子空间”里计算
        它们的 ridge leverage score（近似：P = G (G + rho I)^-1，lev=diag(P)），
        G = X X^T, X=[x1;...;x4]。
      - 选 leverage 最大的原生 token 作为该网格位置的输出（hard pick）。
      - 若 metric 包含多头（head_dim 可指定），则对所有 head 的 leverage 求和/取均值/取最大。

    forward(metric, target_hw) -> prune_fn
    prune_fn(x, mode='pick'|'nearest'|'bilinear')
    """
    def __init__(self,
                 head_dim: int | None = None,   # 若 metric 是 K/V 拼在通道上，传入每头维度，如 64
                 agg: str = 'sum',              # 多头聚合：'sum' | 'mean' | 'max'
                 rho: float = 1e-3,             # ridge 正则，稳定 4x4 逆
                 gamma: float = 0.3,            # 几何 bias（base_logits）权重
                 gate_tau: float | None = 0.02, # 置信度不足时退化为 pure nearest
                 align_corners: bool = False,
                 requires_grad_plan: bool = False):
        super().__init__()
        assert agg in ('sum', 'mean', 'max')
        self.head_dim = head_dim
        self.agg = agg
        self.rho = rho
        self.gamma = gamma
        self.gate_tau = gate_tau
        self.align_corners = align_corners
        self.requires_grad_plan = requires_grad_plan

    def forward(self, metric: torch.Tensor, target_hw: tuple[int, int]):
        """
        metric: [B,C,H,W]
            - 强烈建议：传入“注意力相关”的特征，如第一层/前几层的 K 或 V。
              若传入的是普通特征，也能跑，但收益可能小一些。
            - 若已知每头维度 head_dim（如 64），且 C % head_dim == 0，会当作多头处理。

        target_hw: (H_out, W_out) 例如 (46, 46)
        """
        assert metric.dim() == 4, "metric 必须是 [B,C,H,W]"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        device = metric.device

        # 1) 输出网格 + 2x2 邻域
        Yc, Xc = _make_frac_grid(H, W, H_out, W_out,
                                 self.align_corners, device=device)
        y0, x0, oy, ox = _neighbors_2x2(Yc, Xc)   # K=4
        K = 4
        ys = (y0[..., None] + oy.view(1,1,K)).clamp(0, H-1)  # [H_out,W_out,4]
        xs = (x0[..., None] + ox.view(1,1,K)).clamp(0, W-1)  # [H_out,W_out,4]
        flat_idx = (ys * W + xs).view(-1)                    # [H_out*W_out*4]

        # 2) 几何基权重（仅作 bias 或退化用）
        base_w = _bilinear_base_weights(Yc, Xc, y0, x0).clamp_min(1e-12)  # [H_out,W_out,4]
        base_logits = base_w.log()                                        # [H_out,W_out,4]

        # 3) 准备 metric 作为“注意力子空间”的向量
        #    若提供 head_dim 且可整除，则按多头 [B, nH, d, H, W] 处理
        if (self.head_dim is not None) and (C % self.head_dim == 0):
            nH = C // self.head_dim
            d  = self.head_dim
            met = metric.view(B, nH, d, H, W)           # [B,nH,d,H,W]
            met_flat = met.view(B, nH, d, H*W)          # [B,nH,d,HW]
            # 取 2x2 候选，得到 [B,nH,d,H_out*W_out*4]
            neigh = met_flat[:, :, :, flat_idx].view(B, nH, d, H_out, W_out, K)
            # [B,nH,d,H_out,W_out,4] -> [B,H_out,W_out,4,nH,d]
            neigh = neigh.permute(0, 3, 4, 5, 1, 2).contiguous()
        else:
            # 单头退化：nH=1, d=C
            nH, d = 1, C
            met_flat = metric.view(B, C, H*W)           # [B,C,HW]
            neigh = met_flat[:, :, flat_idx].view(B, C, H_out, W_out, K)
            neigh = neigh.permute(0, 2, 3, 4, 1).unsqueeze(-2).contiguous()
            # 变成 [B,H_out,W_out,4,nH=1,d=C]

        # 4) 计算 ridge leverage scores（对每个 head 独立）
        #    对于每个 head：X=[x1..x4] ∈ R^{4×d},  G = X X^T ∈ R^{4×4}
        #    lev ≈ diag( G (G + rho I)^-1 )
        I4 = torch.eye(4, device=device).view(1,1,1,4,4)
        lev_list = []
        for h in range(nH):
            Xh = neigh[..., h, :]                               # [B,H_out,W_out,4,d]
            # G = X X^T
            G = torch.einsum('bhwkd,bhwmd->bhwkm', Xh, Xh)      # [B,H_out,W_out,4,4]
            A = G + self.rho * I4                               # [B,H_out,W_out,4,4]
            A_inv = torch.linalg.inv(A)                         # [B,H_out,W_out,4,4]
            P = torch.matmul(G, A_inv)                          # [B,H_out,W_out,4,4]
            lev_h = torch.diagonal(P, dim1=-2, dim2=-1)         # [B,H_out,W_out,4]
            lev_list.append(lev_h)

        lev = torch.stack(lev_list, dim=-1)                     # [B,H_out,W_out,4,nH]
        if self.agg == 'sum':
            lev = lev.sum(dim=-1)
        elif self.agg == 'mean':
            lev = lev.mean(dim=-1)
        else: # 'max'
            lev = lev.max(dim=-1).values                        # [B,H_out,W_out,4]

        # 5) 几何 bias + gate（低置信时退为 pure nearest）
        base_logits_b = base_logits.unsqueeze(0).expand(B, -1, -1, -1)  # [B,H_out,W_out,4]
        logits = lev + self.gamma * base_logits_b                        # [B,H_out,W_out,4]

        if self.gate_tau is not None:
            disp = logits.max(dim=-1).values - logits.mean(dim=-1)       # [B,H_out,W_out]
            gate = (disp >= self.gate_tau).float()[..., None]            # [B,H_out,W_out,1]
            logits = gate * logits + (1.0 - gate) * base_logits_b

        # 规划是否参与反传
        logits_for_pick = logits if self.requires_grad_plan else logits.detach()
        base_w_for_bili = base_w if self.requires_grad_plan else base_w.detach()

        # 绑定必要闭包变量
        ys_b, xs_b = ys, xs
        def prune_fn(x: torch.Tensor, mode: str = 'pick') -> torch.Tensor:
            """
            x: [B,C,H,W]  —— 被下采样的特征，H,W 必须与 metric 一致
            mode:
              - 'pick'     : LANS-Down（推荐）
              - 'nearest'  : 纯几何最近邻（2x2 内 base_w 最大者）
              - 'bilinear' : 2x2 bilinear 加权平均（仅作对照）
            """
            assert x.dim() == 4 and x.shape[0] == B and x.shape[2:] == (H, W)
            Bx, Cx = x.shape[0], x.shape[1]

            # gather 2x2 候选实际值
            neigh_vals = []
            for k in range(K):
                vk = x[:, :, ys_b[..., k], xs_b[..., k]]        # [B,C,H_out,W_out]
                neigh_vals.append(vk.permute(0,2,3,1))          # [B,H_out,W_out,C]
            neigh_vals = torch.stack(neigh_vals, dim=3)          # [B,H_out,W_out,4,C]

            if mode == 'pick':
                idx = logits_for_pick.argmax(dim=-1)             # [B,H_out,W_out]
                b_idx = torch.arange(Bx, device=x.device)[:,None,None]
                y_idx = torch.arange(H_out, device=x.device)[None,:,None]
                x_idx = torch.arange(W_out, device=x.device)[None,None,:]
                out = neigh_vals[b_idx, y_idx, x_idx, idx]       # [B,H_out,W_out,C]
                return out.permute(0,3,1,2).contiguous()         # [B,C,H_out,W_out]

            elif mode == 'nearest':
                geo_idx = base_logits_b.argmax(dim=-1)           # [B,H_out,W_out]
                b_idx = torch.arange(Bx, device=x.device)[:,None,None]
                y_idx = torch.arange(H_out, device=x.device)[None,:,None]
                x_idx = torch.arange(W_out, device=x.device)[None,None,:]
                out = neigh_vals[b_idx, y_idx, x_idx, geo_idx]
                return out.permute(0,3,1,2).contiguous()

            elif mode == 'bilinear':
                w = base_w_for_bili / base_w_for_bili.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [H_out,W_out,4]
                w = w.unsqueeze(0)[..., None]                               # [1,H_out,W_out,4,1]
                out = (w * neigh_vals).sum(dim=3)                           # [B,H_out,W_out,C]
                return out.permute(0,3,1,2).contiguous()

            else:
                raise ValueError(f"未知 mode: {mode}")

        prune_fn.info = {
            "H_in": H, "W_in": W, "H_out": H_out, "W_out": W_out,
            "kernel": 2, "K": 4,
            "head_dim": self.head_dim, "agg": self.agg,
            "rho": self.rho, "gamma": self.gamma,
            "gate_tau": self.gate_tau, "align_corners": self.align_corners,
            "requires_grad_plan": self.requires_grad_plan,
            "type": "LANSDownPlanBCHW",
        }
        return prune_fn
