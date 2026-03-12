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
    return torch.stack([w00, w01, w10, w11], dim=-1)  # [H_out,W_out,4]


def _gaussian_base_weights(Yc, Xc, ys, xs, sigma=0.5):
    dy2 = (Yc[..., None] - ys.float()) ** 2
    dx2 = (Xc[..., None] - xs.float()) ** 2
    return torch.exp(-(dy2 + dx2) / (2 * (sigma ** 2)))  # [H_out,W_out,K]


class FlexiToMePlanBCHWv2(nn.Module):
    """
    任意尺度的“语义感知重采样/剪枝”规划器（BCHW）。

    - strategy='flex'：
        基本等价于你原来的设计，几何 + 语义 soft 权重，可用于端到端微调。
    - strategy='semantic_nearest'：
        只在选哪个邻域 token 时用语义（tie-break），
        输出始终是拷贝某个原生 token，更适合 training-free。

    requires_grad_plan=True: 规划阶段可导（默认，便于端到端微调）
    requires_grad_plan=False: 规划阶段权重在闭包中 detach（省显存/更稳）
    """
    def __init__(self,
                 kernel=2,
                 distance='cosine',
                 weighting='softmax',
                 alpha=1.0,
                 beta=1.0,
                 pos_lambda=0.0,
                 gaussian_sigma=0.5,
                 gate_tau=None,
                 align_corners=False,
                 requires_grad_plan=True,
                 strategy: str = "semantic_nearest"  # 默认偏向更安全的策略
                 ):
        super().__init__()
        self.kernel = kernel
        self.distance = distance
        self.weighting = weighting
        self.alpha = alpha
        self.beta = beta
        self.pos_lambda = pos_lambda
        self.gaussian_sigma = gaussian_sigma
        self.gate_tau = gate_tau
        self.align_corners = align_corners
        self.requires_grad_plan = requires_grad_plan
        assert strategy in ("flex", "semantic_nearest"), "strategy 只能是 'flex' 或 'semantic_nearest'"
        self.strategy = strategy

    def forward(self, metric: torch.Tensor, target_hw: tuple[int, int], pos: torch.Tensor = None):
        assert metric.dim() == 4, "metric 必须是 [B,C,H,W]"
        B, C, H, W = metric.shape
        H_out, W_out = target_hw
        device = metric.device

        # 1) 输出网格在输入坐标系的连续坐标
        Yc, Xc = _make_frac_grid(H, W, H_out, W_out, self.align_corners, device=device)

        # 2) 邻域索引
        if self.kernel == 2:
            y0, x0, oy, ox = _neighbors_2x2(Yc, Xc)
        else:
            y0, x0, oy, ox = _neighbors_k(Yc, Xc, self.kernel)

        K = oy.numel()
        ys = (y0[..., None] + oy.view(1, 1, K)).clamp(0, H - 1)  # [H_out,W_out,K]
        xs = (x0[..., None] + ox.view(1, 1, K)).clamp(0, W - 1)  # [H_out,W_out,K]

        # 将 (ys, xs) 编码为单一下标，方便一次 gather
        flat_idx = (ys * W + xs).view(-1)  # [H_out*W_out*K]

        # 3) 几何基权重（base_w）以及 logit（后面统一用 logit 表达）
        if self.kernel == 2:
            base_w = _bilinear_base_weights(Yc, Xc, y0, x0)       # [H_out,W_out,4]
        else:
            base_w = _gaussian_base_weights(Yc, Xc, ys, xs, sigma=self.gaussian_sigma)  # [H_out,W_out,K]

        base_w = base_w.clamp_min(1e-12)                          # 数值稳定
        base_w_pow = base_w ** self.beta                          # [H_out,W_out,K]
        base_logits = base_w_pow.log()                            # [H_out,W_out,K]
        base_w_norm = base_w_pow / base_w_pow.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        base_w_norm = base_w_norm.unsqueeze(0)                    # [1,H_out,W_out,K]

        # 4) 取邻域特征 & 锚点特征（几何插值作为 anchor）
        metric_flat = metric.view(B, C, H * W)                    # [B,C,HW]
        neigh_feats = metric_flat[:, :, flat_idx]                 # [B,C,H_out*W_out*K]
        neigh_feats = neigh_feats.view(B, C, H_out, W_out, K).permute(0, 2, 3, 4, 1)
        # neigh_feats: [B,H_out,W_out,K,C]

        # anchor 用纯几何权重（避免过早引入语义偏移）
        anchor_feat = (base_w_norm[..., None] * neigh_feats).sum(dim=3)  # [B,H_out,W_out,C]

        # 5) 语义相似 + 位置正则
        sim = _similarity(
            neigh_feats,
            anchor_feat.unsqueeze(3).expand_as(neigh_feats),
            distance=self.distance
        )  # [B,H_out,W_out,K]

        if self.pos_lambda > 0 and pos is not None:
            dist2 = (Yc[..., None] - ys.float()) ** 2 + (Xc[..., None] - xs.float()) ** 2  # [H_out,W_out,K]
            sim = sim - self.pos_lambda * dist2.unsqueeze(0)                              # [B,H_out,W_out,K]

        # 6) gate：判断语义是否可靠（disp 小则语义不起作用，退化为 pure base）
        gate = None
        if self.gate_tau is not None:
            disp = sim.max(dim=-1).values - sim.mean(dim=-1)  # [B,H_out,W_out]
            gate = (disp >= self.gate_tau).float()[..., None] # [B,H_out,W_out,1]

        # 7) 组合权重 / logits
        if self.weighting == 'softmax':
            sem_logits = self.alpha * sim                        # [B,H_out,W_out,K]

            if self.strategy == "flex":
                # 完整几何 + 语义 softmax（与原实现等价的 logit 形式）
                logits = sem_logits + base_logits.unsqueeze(0)   # [B,H_out,W_out,K]
                w = torch.softmax(logits, dim=-1)                # [B,H_out,W_out,K]

                if gate is not None:
                    # gate 小时退化到纯几何 base_w_norm
                    pure_base = base_w_norm.expand(B, -1, -1, -1)
                    w = gate * w + (1.0 - gate) * pure_base
            else:
                # semantic_nearest：
                # - 对加权平均等操作：默认只用几何 base_w_norm（不造新 token）
                # - 语义信息只在 pick 模式中做 tie-break，下面在 prune_fn 里用 logits。
                logits = base_logits.unsqueeze(0) + sem_logits    # [B,H_out,W_out,K]
                w = base_w_norm.expand(B, -1, -1, -1)             # [B,H_out,W_out,K]
                if gate is not None:
                    # gate 小时，pick 也强制退化为 pure nearest（下面 prune_fn 会用到 gate）
                    pass
        else:
            # weighting != 'softmax'：退化为纯几何（类似 mean），不引入语义
            logits = base_logits.unsqueeze(0)                     # [1,H_out,W_out,K]
            w = base_w_norm.expand(B, -1, -1, -1)                 # [B,H_out,W_out,K]

        # 规划是否需要参与反传
        if not self.requires_grad_plan:
            w_for_pick = w.detach()
            logits_for_pick = logits.detach()
        else:
            w_for_pick = w
            logits_for_pick = logits

        ys_b, xs_b = ys, xs
        gate_b = gate  # 可能为 None

        def prune_fn(x: torch.Tensor,
                     mode: str = 'wmean',
                     ext_weight: torch.Tensor | None = None) -> torch.Tensor:
            """
            x: [B,C,H,W]
            mode:
                - 'pick'  : 从邻域中选一个 token（hard copy）
                - 'wmean' : 用规划 w 做加权平均
                - 'wsum'  : 同上但不归一化（少见）
                - 'mean'  : 几何平均（退化为插值）
                - 'sum'   : 几何加权和
            """
            assert x.dim() == 4 and x.shape[0] == B and x.shape[2:] == (H, W)
            Bx, Cx, _, _ = x.shape

            # 取邻域值（一次 gather）
            x_flat = x.view(Bx, Cx, H * W)                        # [B,C,HW]
            neigh_vals = x_flat[:, :, flat_idx]                   # [B,C,H_out*W_out*K]
            neigh_vals = neigh_vals.view(Bx, Cx, H_out, W_out, K).permute(0, 2, 3, 4, 1)
            # neigh_vals: [B,H_out,W_out,K,C]

            if mode == 'pick':
                if self.strategy == "semantic_nearest":
                    # 安全策略：始终拷贝原生 token
                    # 基础是 pure nearest 的 base_logits；语义只在 gate 允许时做 tie-break
                    geo_logits = base_logits.unsqueeze(0)         # [1,H_out,W_out,K]
                    pick_logits = geo_logits.expand(B, -1, -1, -1) + self.alpha * sim

                    if gate_b is not None:
                        # gate=0 时只用几何 logit（pure nearest）
                        pick_logits = gate_b * pick_logits + (1.0 - gate_b) * geo_logits

                    pick_logits_use = pick_logits.detach() if not self.requires_grad_plan else pick_logits
                    pick_idx = pick_logits_use.argmax(dim=-1)     # [B,H_out,W_out]
                else:
                    # flex 策略：直接对 w 取 argmax
                    w_pick = w_for_pick
                    pick_idx = w_pick.argmax(dim=-1)              # [B,H_out,W_out]

                # 高维高级索引：从 neigh_vals 中选出对应 token
                b_idx = torch.arange(Bx, device=x.device)[:, None, None]       # [B,1,1]
                y_idx = torch.arange(H_out, device=x.device)[None, :, None]    # [1,H_out,1]
                z_idx = torch.arange(W_out, device=x.device)[None, None, :]    # [1,1,W_out]

                out = neigh_vals[b_idx, y_idx, z_idx, pick_idx]   # [B,H_out,W_out,C]
                return out.permute(0, 3, 1, 2).contiguous()       # [B,C,H_out,W_out]

            # --------- 加权平均 / 加权和 ---------
            if mode in ('mean', 'sum'):
                # 仅几何权重
                w_use = base_w_norm.expand(Bx, -1, -1, -1)
            elif mode in ('wmean', 'wsum'):
                # 使用规划出的语义+几何权重
                w_use = w
            else:
                raise ValueError(f"未知 mode: {mode}")

            if not self.requires_grad_plan:
                w_use = w_use.detach()

            # 叠加外部权重，可导到 ext_weight
            if ext_weight is not None:
                assert ext_weight.shape == (Bx, 1, H, W)
                ew = ext_weight.view(Bx, 1, H * W)[:, :, flat_idx]          # [B,1,H_out*W_out*K]
                ew = ew.view(Bx, 1, H_out, W_out, K).squeeze(1)            # [B,H_out,W_out,K]
                w_use = w_use * ew

            if mode in ('mean', 'wmean'):
                w_use = w_use / w_use.sum(dim=-1, keepdim=True).clamp_min(1e-6)

            out = (w_use[..., None] * neigh_vals).sum(dim=3)      # [B,H_out,W_out,C]
            return out.permute(0, 3, 1, 2).contiguous()           # [B,C,H_out,W_out]

        prune_fn.info = {
            "H_in": H, "W_in": W, "H_out": H_out, "W_out": W_out,
            "kernel": self.kernel, "K": K,
            "distance": self.distance, "weighting": self.weighting,
            "alpha": self.alpha, "beta": self.beta,
            "pos_lambda": self.pos_lambda, "gaussian_sigma": self.gaussian_sigma,
            "gate_tau": self.gate_tau, "align_corners": self.align_corners,
            "requires_grad_plan": self.requires_grad_plan,
            "strategy": self.strategy,
        }
        return prune_fn


if __name__ == "__main__":
    B, C, H, W = 1, 96, 56, 56
    x = torch.randn(B, C, H, W)
    print("input:", x.shape)

    planner = FlexiToMePlanBCHWv2(
        kernel=2,
        distance='cosine',
        weighting='softmax',
        alpha=0.5,          # 建议 training-free 稍小一点
        beta=1.0,
        pos_lambda=0.0,
        gate_tau=0.05,      # 相似度分散太小则退化为 pure nearest
        align_corners=False,
        requires_grad_plan=False,
        strategy="semantic_nearest",  # <-- 免微调推荐用这个
    )

    prune_fn = planner(metric=x, target_hw=(54, 54))
    out_pick = prune_fn(x, mode='pick')
    print("pick out:", out_pick.shape)

    # 如果后续做微调，可以切到 flex + wmean：
    planner_flex = FlexiToMePlanBCHWv2(
        kernel=2,
        distance='cosine',
        weighting='softmax',
        alpha=1.0,
        beta=1.0,
        pos_lambda=0.0,
        gate_tau=None,
        align_corners=False,
        requires_grad_plan=True,
        strategy="flex",
    )
    prune_fn_flex = planner_flex(metric=x, target_hw=(54, 54))
    out_wmean = prune_fn_flex(x, mode='wmean')
    print("wmean out:", out_wmean.shape)
